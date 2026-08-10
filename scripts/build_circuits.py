from __future__ import annotations

import json
import re
from pathlib import Path

_HERE = Path(__file__).resolve().parents[1]
SRC = _HERE if (_HERE / "results").is_dir() else Path("C:/Users/murar/pdac-circuit")
PAGES = _HERE / "_pages"

STATE = {
    "bivalent_poised": "bivalent, poised",
    "polycomb_repressed": "Polycomb repressed",
    "insulator": "insulator",
    "quiescent": "quiescent",
    None: "not called",
}


def load(rel):
    return json.loads((SRC / rel).read_text(encoding="utf-8"))


def summary_table(circuits):
    rows = []
    for c in circuits:
        rows.append(
            f"<tr><td><strong>{c['circuit']}</strong></td>"
            f"<td class=\"num\">{c['pareto_rank']}</td>"
            f"<td class=\"num\">{c['efficacy']:.3f}</td>"
            f"<td class=\"num\">{c['specificity']:.3f}</td>"
            f"<td class=\"num\">{c['robustness']:.2f}</td>"
            f"<td class=\"num\">{c['safety']:.3f}</td>"
            f"<td class=\"num\">{c['tf_knockdown']:.3f}</td>"
            f"<td>{STATE.get(c.get('chromatin_state'), c.get('chromatin_state'))}</td></tr>"
        )
    return ("<div class=\"table-wrap\">\n<table class=\"data\">\n<thead><tr>"
            "<th>Target</th><th>Pareto front</th><th>Efficacy</th><th>Specificity</th>"
            "<th>Robustness</th><th>Safety</th><th>TF knockdown</th><th>Chromatin state</th>"
            "</tr></thead>\n<tbody>\n" + "\n".join(rows) + "\n</tbody>\n</table>\n</div>")


def construct_card(r):
    p, e, g = r["promoter"], r["enhancer"], r["repressor_guide"]
    so, im = r["sequence_optimization"], r["immunogenicity"]
    loc = e["locus"]
    unsat = so.get("unsatisfied") or []
    note = unsat[0] if unsat else "all constraints satisfied"
    rows = [
        ("Target transcription factor", r["target_tf"]),
        ("Subtype", r["subtype"]),
        ("Promoter strength, predicted", f"{p['strength']:.4f}"),
        ("Promoter conformal interval",
         f"[{p['conformal'][0]:.3f}, {p['conformal'][1]:.3f}]"),
        ("Enhancer activity, predicted", f"{e['activity']:.4f}"),
        ("Enhancer measured signal", f"{e['signal']:.4f}"),
        ("Enhancer locus", f"{loc['chrom']}:{loc['start']:,}\u2013{loc['end']:,}"),
        ("CRISPRi protospacer", f"<code>{g['protospacer']}</code> {g['pam']} ({g['strand']})"),
        ("Guide on-target, predicted", f"{g['on_target']:.4f}"),
        ("Guide on-target interval",
         f"[{g['on_conf'][0]:.3f}, {g['on_conf'][1]:.3f}]"),
        ("CFD specificity, genome-wide", f"{g['cfd_specificity']:.3f}"),
        ("Off-targets at or below four mismatches", f"{g['n_off_targets']}"),
        ("GC content, before and after optimisation",
         f"{so['gc_before']:.4f} \u2192 {so['gc_after']:.4f} over {so['n_edits']} edits"),
        ("Immunogenicity proxy",
         f"{im['risk']:.4f}, interval [{im['interval'][0]:.3f}, {im['interval'][1]:.3f}]"),
        ("Composite score, Pareto front", f"{r['composite']:.4f}, front {r['pareto_rank']}"),
    ]
    body = "\n".join(f"<tr><td>{k}</td><td class=\"num\">{v}</td></tr>" for k, v in rows)
    return ("<div class=\"table-wrap\">\n<table class=\"data\">\n"
            "<thead><tr><th>Property</th><th>Value</th></tr></thead>\n"
            f"<tbody>\n{body}\n</tbody>\n</table>\n</div>\n\n"
            f"The optimiser reports {note}")


def campaign_table(rows, k=12):
    out = []
    for r in rows[:k]:
        tgt = r["target"] if r["partner"] is None else f"{r['target']} + {r['partner']}"
        out.append(
            f"<tr><td><strong>{tgt}</strong></td><td>{r['logic']}</td>"
            f"<td class=\"num\">{r['composite']:.4f}</td>"
            f"<td class=\"num\">{r['efficacy']:.3f}</td>"
            f"<td class=\"num\">{r['specificity']:.3f}</td>"
            f"<td class=\"num\">{r['safety']:.3f}</td>"
            f"<td class=\"num\">{r['knockdown']:.3f}</td>"
            f"<td class=\"num\">{r['promoter_strength']:.3f}</td>"
            f"<td class=\"num\">{r['cfd_specificity']:.3f}</td>"
            f"<td><code>{r['protospacer']}</code></td></tr>"
        )
    return ("<div class=\"table-wrap\">\n<table class=\"data\">\n<thead><tr>"
            "<th>Target</th><th>Logic</th><th>Composite</th><th>Efficacy</th><th>Specificity</th>"
            "<th>Safety</th><th>Knockdown</th><th>Promoter</th><th>CFD spec.</th><th>Protospacer</th>"
            "</tr></thead>\n<tbody>\n" + "\n".join(out) + "\n</tbody>\n</table>\n</div>")


def campaign_total():
    p = SRC / "results" / "circuit_design_campaign.json"
    if not p.exists():
        return "a larger set of circuits"
    n = json.loads(p.read_text(encoding="utf-8"))["n_circuits"]
    return f"{n:,} circuits"


def campaign_section():
    p = SRC / "results" / "circuit_design_campaign.json"
    if not p.exists():
        return ""
    s = json.loads(p.read_text(encoding="utf-8"))
    d = s["design_space"]
    rows = s["top_circuits"]
    ext = s.get("extra", {})
    pct = 100.0 * s["n_pareto_front0"] / s["n_circuits"]
    effp = ext.get("pass_efficacy_pct", 0.0)
    return f"""
## The design campaign

The eight circuits below came from one target list with a single best part chosen at every stage. That
is a demonstration rather than a search, so the pipeline was rerun as an enumeration over the part
space. {d['targets_designed']} targets were assembled from the
{d['targets_considered']:,} in the Module I feature matrix, and each was combined against
{d['promoters_per_target']} promoters, up to {d['enhancers_per_target']} enhancer loci and
{d['guides_per_target']} guides. Adding paired-target circuits over
{' and '.join(d['pair_logics'])} logic across the leading {d['pair_top']} targets brings the total to
{s['n_circuits']:,} circuits, of which {s['n_single_target']:,} act on one transcription factor
and {s['n_multi_target']:,} on two. Every one was simulated individually rather than scored from a
formula, which is why the run took {s['runtime_s']/60:.0f} minutes.

The promoters are not eight copies of the same maximum. A pool of {d['gan_pool']:,} sequences was
generated and then sampled across its predicted-strength range, so promoter strength enters as a swept
variable rather than a constant. Enhancer loci are deduplicated before selection and required to sit at
least {d.get('min_locus_separation_bp', 500)} bp apart, because peaks pooled across experiments repeat
the same coordinates and an earlier version of this run was quietly comparing a locus against itself.
Guides are drawn from a {d.get('guide_preshortlist', 48)}-candidate shortlist per target, ranked the way
the pipeline ranks them, on predicted activity multiplied by a local specificity estimate, and all
{s['n_guides_scanned']} survivors carry a genome-wide off-target scan of hg38 at up to four mismatches.

<figure>
  <img src="{{{{ '/images/fig10_design_campaign.png' | relative_url }}}}"
       alt="Four-panel summary of the circuit design campaign">
  <figcaption><b>Figure 11.</b> The design campaign in full. Composite score across the whole
  enumeration separated by single and paired targets, the efficacy against safety trade-off with the
  non-dominated set marked, simulated knockdown against swept promoter strength shown as median and
  interquartile range, and the best circuit found for each target. Generated by
  <code>scripts/make_campaign_figure.py</code> from
  <code>results/circuit_design_campaign_all.jsonl.gz</code>.</figcaption>
</figure>

Composite score runs from {s['composite']['min']:.4f} to {s['composite']['max']:.4f} with a median of
{s['composite']['median']:.4f}, and {s['n_pareto_front0']:,} circuits ({pct:.1f} per cent) are
non-dominated. Simulated knockdown spans {s['knockdown']['min']:.3f} to {s['knockdown']['max']:.3f}.
Every circuit in the enumeration reaches a stable steady state, so stability separates nothing here and
the four objectives carry all of the discrimination.

## The campaign returns a negative

<div class="callout neg">
<p>Not one of the {s['n_circuits']:,} circuits clears the pre-registered floors. {effp:.1f} per cent clear
the efficacy floor of {s['floors']['efficacy_floor']}, and none clears the safety floor of
{s['floors']['safety_floor']}, the best safety score in the whole enumeration being
{ext.get('max_safety', 0):.4f}. Under the thresholds this project registered before running anything, the
enumerated design space contains no acceptable circuit.</p>
</div>

The constraint that binds is guide specificity, and it binds hard. Safety is built from off-target risk,
immunogenicity and integration risk, and off-target risk is one minus CFD specificity, so a guide with a
crowded genome-wide off-target profile caps the safety of every circuit built on it no matter how good the
promoter and enhancer are. Across the {s['n_guides_scanned']} guides that were scanned, the median CFD
specificity is {ext.get('cfd_median', 0):.4f} and the median off-target count at up to four mismatches is
{ext.get('off_median', 0):.0f}. Exactly {ext.get('guides_clearing_minspec', 0)} of {s['n_guides_scanned']}
clears the Module V minimum specificity of {ext.get('min_specificity', 0.5)}, and that one guide, against
{ext.get('best_guide_target', 'its target')}, reaches only {ext.get('max_cfd', 0):.4f}.

This is worth stating plainly rather than hiding behind the {s['n_pareto_front0']:,} non-dominated
circuits. Being non-dominated means nothing else in the set beats a circuit on all four objectives at
once, which is a statement about the geometry of the enumeration. It is not a pass mark, and here the
whole set sits below the bar. The efficacy side of the design problem looks tractable, since most circuits
clear that floor comfortably, while the specificity side does not, and no amount of promoter or enhancer
search repairs it.

One boundary on this negative should be stated. Each target contributed
{d['guides_per_target']} guides to the genome-wide scan, selected from a shortlist of
{d.get('guide_preshortlist', 48)} using a local off-target estimate that only searches the surrounding
few kilobases. A local estimate is a weak proxy for a genome-wide one, so a deeper scan of more guides per
target could surface acceptable candidates that this budget missed. The negative is therefore conditional
on the guide budget, and the honest reading is that acceptable guides are scarce at these loci rather than
proven absent.

### Leading circuits from the enumeration

{campaign_table(rows)}

The full enumeration is committed as `results/circuit_design_campaign_all.jsonl.gz`, one JSON record
per circuit, so the table above can be checked against the complete set rather than taken on trust.

"""


def main():
    cj = load("results/_rac_designed_circuits.json")
    kj = load("results/_rac_designed_constructs.json")
    pay, kpay = cj["payload"], kj["payload"]
    circuits = pay["top_circuits"]
    ranked = {r["target_tf"]: r for r in kpay["ranked_circuits"]}
    vi = kpay["module_VI"]
    top = ranked[circuits[0]["circuit"]]
    called = [c["circuit"] for c in circuits if c.get("chromatin_state")]
    front0 = ", ".join(vi["front0_ids"])
    cav = cj.get("caveats", [])
    araw = next((c for c in cav if "absent from the Module I" in c), "")
    genes = araw.split(":")[-1].strip() if araw else ""
    n_absent = re.search(r"(\d+)", araw)
    words = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five", 6: "Six"}
    nw = words.get(int(n_absent.group(1)), n_absent.group(1)) if n_absent else ""
    absent = (f"{nw} further targets were nominated and could not be designed, namely {genes}, "
              f"because no row for them exists in the Module I feature matrix."
              if genes and n_absent else "")
    sraw = next((c for c in cav if c.startswith("dbSNP")), "")
    m = re.findall(r"(\d+)", sraw)
    snp = (f"A dbSNP cross-check found {m[0]} common variants falling inside the {m[1]} CRISPRi "
           f"windows." if len(m) >= 2 else "")

    body = f"""
A circuit here is a complete, orderable construct rather than a diagram. Each one names a
transcription factor to repress, a promoter to drive the repressor, an enhancer locus that is
accessible in pancreas, a CRISPRi guide placed inside that locus, and the sequence edits needed to
bring the assembled construct into a synthesisable GC band. Two sets are reported. The first is an
enumeration over the part space, in which {campaign_total()} were assembled and individually
simulated. The second is a curated set of {pay['n_circuits']} for the {pay['subtype']} subtype that
predates the enumeration and carries fuller construct detail, {pay['n_pareto_front0']} of them on the
first Pareto front.

Read what follows as engineering output, not as a therapeutic proposal. The scores below are
internal to the design objective and none of them has been measured in a cell.

## Where the targets came from, and why that matters

The target list was produced by the attractor-collapse module, and the claim that attractor collapse
predicts essentiality did not survive review. Collapse failed to beat network degree as a
discriminator, and the essentiality claim was retracted. That retraction is about target
*selection*. It does not touch the design machinery, which takes any target list and returns
constructs, and which is what the numbers on this page describe. A reader who disagrees with the
target list should substitute their own and rerun Module VI; the promoter, enhancer, guide and
optimisation stages are indifferent to how the list was chosen.

{absent} The design stage therefore returns fewer circuits than the target list nominates, which
is the intended behaviour rather than a failure, since a target with no feature row cannot be
scored and the pipeline declines to guess one.

{campaign_section()}## The curated eight

The set below predates the enumeration. It fixes one promoter, one enhancer and one guide per target
and is kept because it is the version carrying complete construct detail, including the sequence
optimisation and immunogenicity fields that the enumeration does not recompute per circuit.

{summary_table(circuits)}

Efficacy, specificity, robustness and safety are the four objectives entering the multi-criteria
decision analysis in Module VI. All {vi['n_circuits']} circuits cleared the pre-registered floors,
and {len(vi['front0_ids'])} occupy the non-dominated front, namely {front0}. Robustness saturates
at 1.00 across the set, which is a property of the stability test rather than a discriminating
result. That test asks whether the simulated steady state survives parameter perturbation, and every
single-transcription-factor negative-feedback topology in this set does survive it. Specificity is where the
circuits actually separate, ranging from {min(c['specificity'] for c in circuits):.3f} to
{max(c['specificity'] for c in circuits):.3f}, and it is also the weakest axis, because
tumour-versus-normal separation is computed across two platforms and is down-weighted for that
reason.

## One circuit in full

The highest composite score belongs to {top['target_tf']}. It is one of {len(called)} targets whose
enhancer carries a called chromatin state, and the only one called bivalent, which is the state most
consistent with a locus that is poised rather than already active.

{construct_card(top)}

The guide sits inside the enhancer window rather than merely near the transcription start site,
which is what makes the construct coherent, since the same locus that supplies the accessibility
evidence also supplies the CRISPRi target. Genome-wide off-target search over hg38 at up to four mismatches
returned no hits, so CFD specificity is reported as 1.000. That figure comes from a substitutions-only
search and does not model bulges, so it is a lower bound on true off-target load rather than a
guarantee.

{snp} Guide selection was made aware of those positions, because a common variant sitting under a
protospacer degrades activity in precisely the donors who carry it, and a guide that works in the
reference genome but not in a quarter of patients is not a usable reagent.

## What the scores do not establish

The efficacy and knockdown figures are simulator output from the ordinary differential equation
model in Module IV, propagated from predicted promoter strength and predicted guide activity. Each
of those inputs carries its own held-out error, and the simulation does not compound them into a
calibrated interval on the final knockdown. The immunogenicity figure is a heuristic proxy and is
labelled as such in the result file. No construct here has been synthesised, transfected, or
assayed. The honest summary is that the pipeline produces internally consistent, fully specified,
synthesisable designs whose predicted behaviour rests on models evaluated on held-out sequence data
and whose biological effect is unmeasured.
""".strip()

    fm = ('---\nlayout: default\ntitle: "Circuits"\n'
          'subtitle: "The eight designed constructs, their parts, and what the scores do and do not mean."\n'
          'description: "The enumerated design space, the leading constructs, and one circuit in full."\n'
          'permalink: /circuits/\n---\n\n')
    PAGES.mkdir(parents=True, exist_ok=True)
    (PAGES / "circuits.md").write_text(fm + body + "\n", encoding="utf-8")
    print(f"  circuits page ({pay['n_circuits']} circuits, front0 {len(vi['front0_ids'])})")


if __name__ == "__main__":
    main()
