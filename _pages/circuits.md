---
layout: default
title: "Circuits"
subtitle: "The eight designed constructs, their parts, and what the scores do and do not mean."
description: "The enumerated design space, the leading constructs, and one circuit in full."
permalink: /circuits/
---

A circuit here is a complete, orderable construct rather than a diagram. Each one names a
transcription factor to repress, a promoter to drive the repressor, an enhancer locus that is
accessible in pancreas, a CRISPRi guide placed inside that locus, and the sequence edits needed to
bring the assembled construct into a synthesisable GC band. Two sets are reported. The first is an
enumeration over the part space, in which a larger set of circuits were assembled and individually
simulated. The second is a curated set of 8 for the classical subtype that
predates the enumeration and carries fuller construct detail, 5 of them on the
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

Four further targets were nominated and could not be designed, namely BRCA2, KMT2C, AGR2, SF3B1, because no row for them exists in the Module I feature matrix. The design stage therefore returns fewer circuits than the target list nominates, which
is the intended behaviour rather than a failure, since a target with no feature row cannot be
scored and the pipeline declines to guess one.

## The curated eight

The set below predates the enumeration. It fixes one promoter, one enhancer and one guide per target
and is kept because it is the version carrying complete construct detail, including the sequence
optimisation and immunogenicity fields that the enumeration does not recompute per circuit.

<div class="table-wrap">
<table class="data">
<thead><tr><th>Target</th><th>Pareto front</th><th>Efficacy</th><th>Specificity</th><th>Robustness</th><th>Safety</th><th>TF knockdown</th><th>Chromatin state</th></tr></thead>
<tbody>
<tr><td><strong>GATA6</strong></td><td class="num">0</td><td class="num">0.915</td><td class="num">0.794</td><td class="num">1.00</td><td class="num">0.877</td><td class="num">0.647</td><td>bivalent, poised</td></tr>
<tr><td><strong>ZNF790</strong></td><td class="num">0</td><td class="num">0.904</td><td class="num">0.581</td><td class="num">1.00</td><td class="num">0.898</td><td class="num">0.574</td><td>not called</td></tr>
<tr><td><strong>SETDB1</strong></td><td class="num">0</td><td class="num">0.954</td><td class="num">0.573</td><td class="num">1.00</td><td class="num">0.887</td><td class="num">0.658</td><td>not called</td></tr>
<tr><td><strong>E2F1</strong></td><td class="num">0</td><td class="num">0.930</td><td class="num">0.511</td><td class="num">1.00</td><td class="num">0.900</td><td class="num">0.610</td><td>not called</td></tr>
<tr><td><strong>SOX13</strong></td><td class="num">0</td><td class="num">0.898</td><td class="num">0.580</td><td class="num">1.00</td><td class="num">0.899</td><td class="num">0.641</td><td>not called</td></tr>
<tr><td><strong>AHR</strong></td><td class="num">1</td><td class="num">0.760</td><td class="num">0.451</td><td class="num">1.00</td><td class="num">0.898</td><td class="num">0.617</td><td>insulator</td></tr>
<tr><td><strong>MYBL2</strong></td><td class="num">1</td><td class="num">0.911</td><td class="num">0.431</td><td class="num">1.00</td><td class="num">0.898</td><td class="num">0.623</td><td>Polycomb repressed</td></tr>
<tr><td><strong>ZNF331</strong></td><td class="num">1</td><td class="num">0.863</td><td class="num">0.497</td><td class="num">1.00</td><td class="num">0.896</td><td class="num">0.606</td><td>quiescent</td></tr>
</tbody>
</table>
</div>

Efficacy, specificity, robustness and safety are the four objectives entering the multi-criteria
decision analysis in Module VI. All 8 circuits cleared the pre-registered floors,
and 5 occupy the non-dominated front, namely GATA6, SETDB1, E2F1, SOX13, MYBL2. Robustness saturates
at 1.00 across the set, which is a property of the stability test rather than a discriminating
result. That test asks whether the simulated steady state survives parameter perturbation, and every
single-transcription-factor negative-feedback topology in this set does survive it. Specificity is where the
circuits actually separate, ranging from 0.431 to
0.794, and it is also the weakest axis, because
tumour-versus-normal separation is computed across two platforms and is down-weighted for that
reason.

## One circuit in full

The highest composite score belongs to GATA6. It is one of 4 targets whose
enhancer carries a called chromatin state, and the only one called bivalent, which is the state most
consistent with a locus that is poised rather than already active.

<div class="table-wrap">
<table class="data">
<thead><tr><th>Property</th><th>Value</th></tr></thead>
<tbody>
<tr><td>Target transcription factor</td><td class="num">GATA6</td></tr>
<tr><td>Subtype</td><td class="num">classical</td></tr>
<tr><td>Promoter strength, predicted</td><td class="num">0.9868</td></tr>
<tr><td>Promoter conformal interval</td><td class="num">[0.000, 1.497]</td></tr>
<tr><td>Enhancer activity, predicted</td><td class="num">0.8007</td></tr>
<tr><td>Enhancer measured signal</td><td class="num">0.4347</td></tr>
<tr><td>Enhancer locus</td><td class="num">chr18:22,168,993–22,170,595</td></tr>
<tr><td>CRISPRi protospacer</td><td class="num"><code>AGAGACTAGAAGTTGGTCCG</code> CGG (-)</td></tr>
<tr><td>Guide on-target, predicted</td><td class="num">0.9704</td></tr>
<tr><td>Guide on-target interval</td><td class="num">[0.515, 1.000]</td></tr>
<tr><td>CFD specificity, genome-wide</td><td class="num">1.000</td></tr>
<tr><td>Off-targets at or below four mismatches</td><td class="num">0</td></tr>
<tr><td>GC content, before and after optimisation</td><td class="num">0.5944 → 0.5900 over 35 edits</td></tr>
<tr><td>Immunogenicity proxy</td><td class="num">0.0254, interval [0.000, 0.275]</td></tr>
<tr><td>Composite score, Pareto front</td><td class="num">0.9127, front 0</td></tr>
</tbody>
</table>
</div>

The optimiser reports gc_out_of_band: window@1980 gc=0.860 outside (0.4, 0.6); feasibility limited by amino-acid composition (no synonymous swap reaches the band)

The guide sits inside the enhancer window rather than merely near the transcription start site,
which is what makes the construct coherent, since the same locus that supplies the accessibility
evidence also supplies the CRISPRi target. Genome-wide off-target search over hg38 at up to four mismatches
returned no hits, so CFD specificity is reported as 1.000. That figure comes from a substitutions-only
search and does not model bulges, so it is a lower bound on true off-target load rather than a
guarantee.

A dbSNP cross-check found 26 common variants falling inside the 8 CRISPRi windows. Guide selection was made aware of those positions, because a common variant sitting under a
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
