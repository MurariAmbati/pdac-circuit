from __future__ import annotations

import html
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "index.html"
REPO = "https://github.com/MurariAmbati/pdac-chromatin-circuit"


def load(rel):
    p = ROOT / rel
    if not p.exists():
        raise FileNotFoundError(f"required source missing: {rel}")
    return json.loads(p.read_text(encoding="utf-8"))


def f(x,n=4):
    return "n/a" if x is None else f"{float(x):.{n}f}"


def i(x):
    return "n/a" if x is None else f"{int(x):,}"


def esc(s):
    return html.escape(str(s))


def line_chart(points,xkey,ykey,*,ylo,yhi,width=560,height=230,label=""):
    pad_l,pad_r,pad_t,pad_b = 54,16,16,34
    iw,ih = width - pad_l - pad_r,height - pad_t - pad_b
    xs = [p[xkey] for p in points]
    ys = [p[ykey] for p in points]
    xlo,xhi = min(xs),max(xs)

    def px(v):
        return pad_l + (0 if xhi == xlo else (v - xlo) / (xhi - xlo)) * iw

    def py(v):
        return pad_t + ih - (v - ylo) / (yhi - ylo) * ih

    grid = []
    for t in range(5):
        v = ylo + (yhi - ylo) * t / 4
        y = py(v)
        grid.append(f'<line class="gl" x1="{pad_l}" y1="{y:.1f}" x2="{width - pad_r}" y2="{y:.1f}"/>')
        grid.append(f'<text class="ax" x="{pad_l - 8}" y="{y + 4:.1f}" text-anchor="end">{v:.2f}</text>')
    path = " ".join(("M" if k == 0 else "L") + f"{px(x):.1f} {py(y):.1f}" for k,(x,y) in enumerate(zip(xs,ys)))
    dots = "".join(f'<circle class="pt" cx="{px(x):.1f}" cy="{py(y):.1f}" r="4"><title>n={x:,} → {y:.4f}</title></circle>'
                   for x,y in zip(xs,ys))
    xlab = "".join(f'<text class="ax" x="{px(x):.1f}" y="{height - 10}" text-anchor="middle">{x // 1000}k</text>'
                   for x in xs)
    return (f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{esc(label)}" class="chart">'
            f'{"".join(grid)}<path class="ln" d="{path}"/>{dots}{xlab}</svg>')


def build():
    pro_m = load("models/promoter.model.json")["metrics"]
    enh_m = load("models/enhancer.model.json")["metrics"]
    grna_m = load("models/grna_ontarget.model.json")["metrics"]
    gan_m = load("models/promoter_gan.model.json")["metrics"]

    pro_s = load("results/promoter_scaleup.json")
    enh_s = load("results/enhancer_scaleup.json")
    grna_s = load("results/grna_cnn_kim_retrain.json")
    grna_a = load("results/grna_kim_augment.json")
    gan_s = load("results/promoter_gan_scaleup.json")
    panc = load("results/enhancer_panc1_augment.json")
    pcurve = load("results/promoter_scaling_curve.json")
    ecurve = load("results/enhancer_scaling_curve.json")
    k27 = load("results/pdac_residual_foldchange_H3K27ac.json")
    rig = load("results/rigorous_validation.json")["A_rac_vs_degree"]
    tvb = k27["targets_vs_all_background"]

    models = [
        ("gRNA on-target","V","GBT + CNN","Spearman",f(grna_m.get("spearman_ensemble")),
         f(grna_s.get("shipped_ensemble")),"Doench-2016 + Kim-2019, 18,142 real HT guides"),
        ("Promoter strength","II","GBT + CNN","Spearman",f(pro_m.get("spearman_ensemble")),
         f(pro_s.get("baseline_shipped_ensemble")),"FANTOM5 CAGE, 209,374 real peaks"),
        ("Enhancer activity","II","multitask CNN","AUROC",f(enh_m.get("auroc")),
         f(enh_s.get("baseline_shipped_auroc")),"ENCODE pancreas ATAC ∩ H3K27ac"),
        ("Promoter generator","VII","WGAN-GP","4-mer JS",f(gan_m.get("js_gen_vs_real")),
         f(gan_s.get("baseline_js_gen_vs_real")),"FANTOM5 top-quartile, 52,342 real promoters"),
    ]
    rows = "".join(
        f"<tr><td><b>{esc(n)}</b></td><td>{esc(mod)}</td><td class='mono'>{esc(a)}</td>"
        f"<td>{esc(met)}</td><td class='num before'>{esc(b)}</td><td class='num after'>{esc(cur)}</td>"
        f"<td class='src'>{esc(src)}</td></tr>"
        for n,mod,a,met,cur,b,src in models)

    ppts = [{"n": p["n_train"],"v": p["ensemble"]} for p in pcurve["points"]]
    epts = [{"n": p["n_train"],"v": p["auroc"]} for p in ecurve["points"]]
    pchart = line_chart(ppts,"n","v",ylo=0.48,yhi=0.55,label="Promoter Spearman versus training-set size")
    echart = line_chart(epts,"n","v",ylo=0.79,yhi=0.83,label="Enhancer AUROC versus training-set size")

    per_target = k27["per_target"]
    top = sorted(per_target.items(),key=lambda kv: -kv[1]["log2_residual"])[:8]
    k27rows = "".join(
        f"<tr><td class='mono'>{esc(g)}</td><td class='num'>{f(v['pdac_fc'],2)}</td>"
        f"<td class='num'>{f(v['healthy_fc'],2)}</td>"
        f"<td class='num {'up' if v['log2_residual'] > 0 else 'dn'}'>{v['log2_residual']:+.3f}</td></tr>"
        for g,v in top)

    css = """
:root{--bg:#fbfbfd;--fg:#16181d;--mut:#5a6272;--line:#e3e6ec;--card:#fff;--acc:#1f6feb;--good:#127a3d;--bad:#b3261e;--warn:#8a5a00;--warnbg:#fff8e6;--code:#f3f5f9}
@media (prefers-color-scheme:dark){:root{--bg:#0d1117;--fg:#e6edf3;--mut:#9aa4b2;--line:#242b36;--card:#151b24;--acc:#589bff;--good:#3fb950;--bad:#f85149;--warn:#d9a441;--warnbg:#1e1806;--code:#111720}}
:root[data-theme=dark]{--bg:#0d1117;--fg:#e6edf3;--mut:#9aa4b2;--line:#242b36;--card:#151b24;--acc:#589bff;--good:#3fb950;--bad:#f85149;--warn:#d9a441;--warnbg:#1e1806;--code:#111720}
:root[data-theme=light]{--bg:#fbfbfd;--fg:#16181d;--mut:#5a6272;--line:#e3e6ec;--card:#fff;--acc:#1f6feb;--good:#127a3d;--bad:#b3261e;--warn:#8a5a00;--warnbg:#fff8e6;--code:#f3f5f9}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);font:16px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,sans-serif;-webkit-font-smoothing:antialiased}
.wrap{max-width:980px;margin:0 auto;padding:0 22px 96px}
header{padding:26px 0 30px;border-bottom:1px solid var(--line)}
h1{font-size:2.35rem;line-height:1.15;margin:0 0 10px;letter-spacing:-.022em}
.sub{color:var(--mut);font-size:1.09rem;max-width:74ch;margin:0}
.ruo{display:inline-block;margin:20px 0 0;padding:5px 11px;border:1px solid var(--warn);color:var(--warn);background:var(--warnbg);border-radius:6px;font-size:.76rem;font-weight:650;letter-spacing:.05em;text-transform:uppercase}
h2{font-size:1.35rem;margin:52px 0 6px;letter-spacing:-.015em;scroll-margin-top:20px}
h3{font-size:1.02rem;margin:30px 0 6px}
.lede{color:var(--mut);margin:0 0 18px;max-width:76ch}
p{max-width:76ch}
table{width:100%;border-collapse:collapse;margin:18px 0;font-size:.9rem}
th,td{text-align:left;padding:9px 11px;border-bottom:1px solid var(--line);vertical-align:top}
th{font-size:.71rem;text-transform:uppercase;letter-spacing:.06em;color:var(--mut);font-weight:650}
tbody tr:hover{background:var(--code)}
.num{text-align:right;font-variant-numeric:tabular-nums;font-family:ui-monospace,SFMono-Regular,Menlo,monospace}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:.87em}
.before{color:var(--mut)}
.after{color:var(--good);font-weight:650}
.up{color:var(--good)}
.dn{color:var(--bad)}
.src{color:var(--mut);font-size:.83em}
.scroll{overflow-x:auto;-webkit-overflow-scrolling:touch}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:14px;margin:22px 0}
.card{background:var(--card);border:1px solid var(--line);border-radius:11px;padding:16px 17px}
.card .k{font-size:1.72rem;font-weight:680;letter-spacing:-.02em;font-variant-numeric:tabular-nums}
.card .l{color:var(--mut);font-size:.79rem;margin-top:3px;line-height:1.45}
.note{border-left:3px solid var(--acc);background:var(--card);padding:13px 17px;border-radius:0 9px 9px 0;margin:20px 0}
.note.bad{border-left-color:var(--bad)}
.note.warn{border-left-color:var(--warn)}
.note b{font-weight:670}
.charts{display:grid;grid-template-columns:repeat(auto-fit,minmax(300px,1fr));gap:22px;margin:20px 0}
.chart{width:100%;height:auto;background:var(--card);border:1px solid var(--line);border-radius:11px}
.gl{stroke:var(--line);stroke-width:1}
.ln{fill:none;stroke:var(--acc);stroke-width:2.4;stroke-linejoin:round;stroke-linecap:round}
.pt{fill:var(--acc)}
.ax{fill:var(--mut);font-size:10px;font-family:ui-monospace,monospace}
.cap{color:var(--mut);font-size:.82rem;margin:-8px 0 0}
code{background:var(--code);padding:1.5px 5px;border-radius:4px;font-size:.87em;font-family:ui-monospace,monospace}
a{color:var(--acc);text-decoration:none}
a:hover{text-decoration:underline}
footer{margin-top:62px;padding-top:22px;border-top:1px solid var(--line);color:var(--mut);font-size:.85rem}
ul{max-width:76ch}
li{margin:5px 0}
.bar{display:flex;justify-content:flex-end;padding-top:16px}
.toggle{background:var(--card);border:1px solid var(--line);color:var(--mut);border-radius:8px;padding:6px 11px;font-size:.78rem;cursor:pointer;font-family:inherit}
.toggle:hover{color:var(--fg);border-color:var(--mut)}
@media(max-width:640px){h1{font-size:1.8rem}header{padding:14px 0 24px}.wrap{padding:0 16px 64px}}
"""

    js = ("(function(){var r=document.documentElement,b=document.getElementById('tg');"
          "function s(t){r.setAttribute('data-theme',t);try{localStorage.setItem('th',t)}catch(e){}}"
          "try{var v=localStorage.getItem('th');if(v)s(v)}catch(e){}"
          "b.addEventListener('click',function(){var d=r.getAttribute('data-theme');"
          "if(!d)d=matchMedia('(prefers-color-scheme:dark)').matches?'dark':'light';"
          "s(d==='dark'?'light':'dark')})})();")

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PDAC Chromatin-Circuit — computational results</title>
<meta name="description" content="A computational PDAC synthetic gene-circuit prototype and auditing framework. Real-data results, honest negatives, and retractions.">
<style>{css}</style>
</head>
<body>
<div class="wrap">
<div class="bar"><button class="toggle" id="tg" type="button">theme</button></div>

<header>
<h1>PDAC Chromatin-Circuit</h1>
<p class="sub">A computational prototype and auditing framework for designing synthetic gene circuits against
pancreatic-ductal-adenocarcinoma transcription factors, built entirely on real public data. This page reports what
survived validation and what did not.</p>
<div class="ruo">Research use only — not for clinical or diagnostic use</div>
</header>

<div class="note bad">
<b>Read this first.</b> This is <b>not</b> a validated therapeutic platform. It has produced no wet-lab result, no
cloning-ready construct, and no safe final guide RNA. One biological finding survived adversarial validation
(H3K27ac, below). The project's original headline claim was <b>retracted</b> after a head-to-head test it had never
been given. The end-to-end circuit pipeline currently <b>abstains</b> rather than emit designs built on numbers it
cannot stand behind.
</div>

<h2 id="models">Trained models</h2>
<p class="lede">Four learned sequence models, each trained from scratch on real public data with a
leakage-controlled held-out split. Every "current" figure below is read directly from the deployed model manifest
at build time, so this page cannot drift from the shipped weights.</p>
<div class="scroll">
<table>
<thead><tr><th>Model</th><th>Module</th><th>Architecture</th><th>Metric</th><th class="num">Before</th><th class="num">Current</th><th>Real training data</th></tr></thead>
<tbody>{rows}</tbody>
</table>
</div>
<p class="cap">gRNA and promoter/enhancer report Spearman and AUROC (higher is better). The generator reports
4-mer Jensen–Shannon divergence against real promoters (<b>lower</b> is better); random DNA scores
{f(gan_s.get('scaleup_js_random_vs_real'))} on the same reference.</p>

<h2 id="scaling">The data-scaling programme</h2>
<p class="lede">Every shipped model turned out to be limited by an arbitrary cap on how much of its own real data it
was allowed to see, not by its features or architecture. Removing those caps is the bulk of the current results.
Each comparison is apples-to-apples: the previous model is re-scored on the identical held-out set, and ensemble
weights are chosen on a validation split that never touches test.</p>

<div class="cards">
<div class="card"><div class="k">{i(grna_s.get('n_train'))}</div><div class="l">gRNA training guides, up from {i(5310)} across only 17 genes</div></div>
<div class="card"><div class="k">{i(pro_s.get('n_train'))}</div><div class="l">promoter training peaks, up from a 60,000 cap</div></div>
<div class="card"><div class="k">{i(enh_s.get('n_train'))}</div><div class="l">enhancer training rows, up from a 20,000-active cap</div></div>
<div class="card"><div class="k">{i(gan_s.get('n_train_uncapped'))}</div><div class="l">real promoters for the generator, up from a 12,000 cap</div></div>
</div>

<h3>gRNA on-target: adding a second real dataset</h3>
<p>The model was feature-saturated but data-starved. Adding <b>Kim et al. 2019</b> ({i(grna_s.get('n_kim_in_train'))}
high-throughput SpCas9 guides) lifted held-out Spearman from {f(grna_s.get('shipped_ensemble'))} to
<b>{f(grna_s.get('deployed_ensemble'))}</b> on the same {i(grna_s.get('n_test'))} held-out-gene guides. The decisive
check came <em>before</em> merging: training on Doench's 17 genes and testing on Kim scored
{f(grna_a.get('cross_dataset_doench_to_kim'))} — higher than the within-Doench held-out — showing the model had
learned transferable guide biology rather than memorising 17 genes. The CNN carries the gain, going from
{f(grna_s.get('cnn_doench_only_prev'))} to <b>{f(grna_s.get('cnn_doench_plus_kim'))}</b>: it had been a near-random
component down-weighted to survive, and is now a genuine contributor.</p>

<h3>Promoter and enhancer: scaling curves</h3>
<div class="charts">
<div>{pchart}<p class="cap">Promoter ensemble Spearman versus training peaks, fixed chr8/chr9 test
({i(pcurve.get('n_test'))} peaks).</p></div>
<div>{echart}<p class="cap">Enhancer AUROC versus training rows, fixed chr8/chr9 test
({i(ecurve.get('n_test'))} rows).</p></div>
</div>
<p>The promoter curve rises monotonically across an 18-fold increase in real data and then flattens, which is what
data-limited-then-saturating looks like; the enhancer curve is flatter throughout, consistent with a model already
close to its ceiling. Both gains are real but modest, and reporting them as such is the point.</p>

<h2 id="surviving">The one surviving biological result</h2>
<p class="lede">H3K27ac enrichment at the promoters of prioritised target transcription factors, PDAC versus healthy
pancreas, measured on ENCODE fold-change-over-control tracks.</p>
<div class="cards">
<div class="card"><div class="k up">{tvb['target_mean_log2']:+.3f}</div><div class="l">mean log2 residual across {i(tvb['n_targets'])} targets</div></div>
<div class="card"><div class="k">{tvb['background_mean_log2']:+.3f}</div><div class="l">mean across {i(tvb['n_background'])} background loci</div></div>
<div class="card"><div class="k">{tvb['mannwhitney_p_greater']:.4f}</div><div class="l">Mann–Whitney p, one-sided</div></div>
<div class="card"><div class="k">{tvb['target_frac_up']:.0%}</div><div class="l">of targets up, versus {tvb['background_frac_up']:.0%} of background</div></div>
</div>
<div class="scroll">
<table>
<thead><tr><th>Target</th><th class="num">PDAC fold-change</th><th class="num">Healthy fold-change</th><th class="num">log2 residual</th></tr></thead>
<tbody>{k27rows}</tbody>
</table>
</div>
<p class="cap">Eight strongest of {i(tvb['n_targets'])} targets. Verdict recorded in the result file:
<b>{esc(k27['verdict'])}</b>.</p>
<div class="note warn">
<b>Limits of this result, stated plainly.</b> It rests on a <b>single PDAC cell line</b> (PANC-1) against one healthy
fold-change track per mark, where the earlier run averaged up to six. The healthy reference is therefore noisier.
The effect is roughly 1.5–1.8 fold and <b>ATAC accessibility does not replicate it</b>. It is a hypothesis worth
testing in primary tumours, not an established fact.
</div>

<h2 id="retracted">Retracted and negative results</h2>
<p class="lede">These are reported with the same prominence as the positive ones. Most of this project's honest value
is here.</p>

<h3>Retracted: regulatory-attractor control predicts essentiality</h3>
<p>The original claim was that a bistable attractor model's collapse score predicts held-out CRISPR essentiality and
beats network centrality. Forced into a head-to-head test it had never been given, it did not survive.</p>
<div class="scroll">
<table>
<thead><tr><th>Predictor</th><th class="num">AUC</th><th class="num">PR-AUC</th></tr></thead>
<tbody>
<tr><td>Attractor collapse</td><td class="num">{f(rig['auc_rac'])}</td><td class="num">{f(rig['pr_auc_rac'])}</td></tr>
<tr><td><b>Network degree</b></td><td class="num after">{f(rig['auc_degree'])}</td><td class="num after">{f(rig['pr_auc_degree'])}</td></tr>
<tr><td>Eigenvector centrality</td><td class="num">{f(rig['auc_eigenvector'])}</td><td class="num">—</td></tr>
</tbody>
</table>
</div>
<p>Δ AUC versus degree is <b>{f(rig['delta_auc_rac_minus_degree'])}</b>, 95% CI
[{f(rig['delta_auc_ci95_paired_bootstrap'][0])}, {f(rig['delta_auc_ci95_paired_bootstrap'][1])}],
p = {f(rig['delta_auc_p_two_sided'],3)}. Controlling for degree, expression and variance, the partial Spearman is
{f(rig['covariate_control']['partial_spearman_collapse_vs_essentiality_given_degree_expr_var'][0])}
(p = {f(rig['covariate_control']['partial_spearman_collapse_vs_essentiality_given_degree_expr_var'][1],3)}), and adding
collapse to a covariates-only model moves cross-validated AUC from
{f(rig['covariate_control']['auc_covariates_only_cv'])} to
{f(rig['covariate_control']['auc_covariates_plus_collapse_cv'])} — that is, nothing. On
{i(rig['n_genes'])} genes with {i(rig['n_positive'])} positives, <b>collapse adds no information beyond network
degree.</b> The earlier figure came from selecting on the same labels used to report it. The bistable formulation
stands only as intervention semantics, not as a predictor.</p>

<h3>Negative: adding PDAC chromatin to the enhancer model</h3>
<p>A pancreas-only enhancer model already predicts PANC-1 PDAC enhancers at AUROC
<b>{f(panc['pancreas_only_panc1_test_xdomain'])}</b>, above its own pancreas test
({f(panc['pancreas_only_pancreas_test'])}), and the reverse direction holds at
{f(ecurve.get('reverse_xdomain_panc1_to_pancreas'))}. The grammar is genuinely shared. Merging PANC-1 into training
therefore did not help the benchmark ({f(panc['merged_pancreas_test'])},
{panc['delta_pancreas_test']:+.4f}) and was <b>not deployed</b>.</p>

<h3>Negative: un-capping the generator did not improve realism</h3>
<p>Removing the generator's data cap did not lower 4-mer divergence
({f(gan_s['baseline_js_gen_vs_real'])} → {f(gan_s['scaleup_js_gen_vs_real'])}); that axis was already near its floor.
It did strengthen the selectable tail the pipeline actually consumes
({f(gan_s['baseline_p90'])} → <b>{f(gan_s['scaleup_p90'])}</b> 90th-percentile predicted strength), which is why the
un-capped generator is deployed. Both versions clear the pre-registered certification.</p>

<h3>The pipeline abstains</h3>
<p>After the off-target search was repaired to cover the genome rather than a locus neighbourhood, no candidate guide
clears genome-wide specificity. The end-to-end run therefore returns
<b>{esc(load('results/run_classical_fixed.json')['verdict'])}</b> with certification
<code>{esc(load('results/run_classical_fixed.json')['cert'])}</code> and emits zero circuits. That is the correct
outcome: absence of a search is not evidence of specificity.</p>

<h2 id="repro">Reproducibility</h2>
<ul>
<li><b>Real data only.</b> Every corpus is recorded in <code>data/manifests/</code> with source URL, byte count and
sha256. Raw bytes stay out of git; the hashes are the provenance.</li>
<li><b>Frozen predeploy fixtures.</b> Each deployed model ships a frozen set of real test rows and their CPU
predictions; reloading the checkpoint must reproduce them to 1e-4 or the model fails its gate.</li>
<li><b>Leakage-controlled splits.</b> Gene-grouped for gRNA; chromosome-held-out (test chr8/chr9, validation chr7)
for promoter and enhancer. Ensemble weights are selected on validation, never on test.</li>
<li><b>Pre-registration.</b> Thresholds and margins live in the registry before training, so a result either clears a
pre-committed bar or is reported as not clearing it.</li>
</ul>

<h2 id="reading">Full write-ups</h2>
<ul>
<li><a href="{REPO}/blob/master/docs/ADDENDUM_DATA_SCALING.md">Data-scaling programme</a> — the four models, scaling curves, cross-domain transfer</li>
<li><a href="{REPO}/blob/master/RESULTS.md">RESULTS.md</a> — the complete result ledger</li>
<li><a href="{REPO}/blob/master/COMPENDIUM.md">COMPENDIUM.md</a> — methods and full compilation</li>
<li><a href="{REPO}/blob/master/AUDIT_RESPONSE.md">AUDIT_RESPONSE.md</a> — disposition of every external-audit finding</li>
<li><a href="{REPO}/blob/master/docs/ADDENDUM_CHROMATIN.md">Chromatin addendum</a> — the H3K27ac result and its controls</li>
</ul>

<footer>
<p>Research use only. Generated from the repository's result files by <code>scripts/build_site.py</code>, so the
figures on this page are the figures in the data. Source:
<a href="{REPO}">{esc(REPO.replace('https://github.com/',''))}</a>.</p>
</footer>

</div>
<script>{js}</script>
</body>
</html>
"""
    OUT.parent.mkdir(parents=True,exist_ok=True)
    OUT.write_text(doc,encoding="utf-8")
    (ROOT / "docs" / ".nojekyll").write_text("",encoding="utf-8")
    print(f"wrote {OUT.relative_to(ROOT)} ({len(doc):,} bytes)")


if __name__ == "__main__":
    build()
