---
layout: default
title: Validation
subtitle: The claims that did not survive, reported with the same prominence as the ones that did.
permalink: /validation/
---

Most of this project's value is on this page. A pipeline that only reports its successes is not measuring
anything; the tests below were designed to break the project's own results, and several of them did.

## 1. Retracted — attractor collapse predicts essentiality

**The original claim.** Module VIII modelled the PDAC regulatory network as a bistable dynamical system
`x ← σ(gain · (Wx + b))`, fit so that real cell states form the viable attractor. Clamping a gene down and
measuring the network-wide collapse toward the dead attractor gave a "collapse score", which was reported to
predict held-out CRISPR essentiality at AUC 0.653 with a 50,000-permutation p = 0.0022, beating network
degree (0.629) and eigenvector centrality (0.584).

**The test that had never been run.** The permutation test compared collapse against *chance*. It never
compared collapse against the baseline it claimed to beat. A direct head-to-head, with a paired bootstrap over
the same genes, gives a different answer.

<figure>
  <img src="{{ '/images/fig5_rac_validation.png' | relative_url }}" alt="Attractor collapse versus degree and eigenvector centrality, with a confidence interval on the difference">
  <figcaption><b>Figure 7.</b> Across 419 genes with 31 essential positives, attractor collapse reaches AUC
  0.547 against network degree's 0.629. The paired-bootstrap 95% confidence interval on the difference spans
  zero.</figcaption>
</figure>

<div class="tablewrap">
<table>
<thead><tr><th>Statistic</th><th class="num">Value</th></tr></thead>
<tbody>
<tr><td>AUC, attractor collapse</td><td class="num dn">0.5471</td></tr>
<tr><td>AUC, network degree</td><td class="num up">0.6290</td></tr>
<tr><td>AUC, eigenvector centrality</td><td class="num">0.5843</td></tr>
<tr><td>Δ AUC (collapse − degree)</td><td class="num dn">−0.0819</td></tr>
<tr><td>95% CI, paired bootstrap</td><td class="num">[−0.199, +0.029]</td></tr>
<tr><td>p, two-sided</td><td class="num">0.147</td></tr>
<tr><td>PR-AUC, collapse vs degree</td><td class="num">0.098 vs 0.131</td></tr>
<tr><td>Partial Spearman given degree, expression, variance</td><td class="num">0.028 (p = 0.563)</td></tr>
<tr><td>Cross-validated AUC, covariates only → plus collapse</td><td class="num">0.653 → 0.652</td></tr>
</tbody>
</table>
</div>

**Verdict: retracted.** Collapse adds nothing beyond network degree. Controlling for degree, expression and
variance, its partial correlation with essentiality is 0.028 with p = 0.56, and adding it to a covariates-only
model moves cross-validated AUC by −0.001. The original 0.653 came from selecting the model configuration on
the same CRISPR labels used to report it.

A separate dynamics audit went further: the fitted system is **not bistable at any gain**, so the mechanism
the claim rested on does not exist as described. The formulation survives only as a way of expressing
interventions, not as a predictor.

## 2. Negative — adding PDAC chromatin to the enhancer model

A genuinely new real dataset does not automatically improve a model. Adding PANC-1 PDAC ATAC and H3K27ac to
the healthy-pancreas enhancer training set was tested under the same controlled protocol as every other
data addition.

<div class="tablewrap">
<table>
<thead><tr><th>Training set</th><th class="num">Pancreas test</th><th class="num">PANC-1 test</th></tr></thead>
<tbody>
<tr><td>Pancreas only</td><td class="num up">0.8150</td><td class="num">0.8349</td></tr>
<tr><td>Pancreas + PANC-1</td><td class="num dn">0.8096</td><td class="num">0.8306</td></tr>
</tbody>
</table>
</div>

**Verdict: not deployed.** Merging costs 0.0054 AUROC on the benchmark. The reason is visible in the same
table — the pancreas-only model already scores 0.835 on PANC-1, better than on its own domain. The model was
never data-limited on enhancer *sequence*; it was limited on pancreatic sample count. Mixing domains
therefore dilutes rather than adds. This is worth recording precisely because the cross-domain result
explains the negative.

## 3. Negative — un-capping the generator did not improve realism

Removing the generator's 12,000-promoter cap and retraining on all 52,342 did not lower 4-mer divergence:
0.0088 → 0.0123, slightly worse and within run-to-run variation for a GAN whose divergence already sat near
its floor.

It did substantially strengthen the selectable tail — 90th-percentile predicted strength 0.937 → 0.992, median
uplift −0.021 → +0.106 — which is the axis the pipeline actually consumes, since it selects the strongest
generated promoter from a library. The un-capped generator is deployed on that basis, with the realism cost
stated. Both versions clear the pre-registered certification.

Because the generator trains only on real sequence and never against the promoter model, the stronger tail is
a genuine learned property rather than an artefact of optimising the scorer.

## 4. Retracted — off-target risk of 0.00

Guides were originally reported with an off-target risk of 0.00, which came from searching a window around
the target locus covering roughly 0.001% of the genome. That is not a specificity measurement.

Rescored with the exact Doench-2016 CFD nucleotide-pair matrix over a genome-wide scan, **all four final
guides fail** the pre-registered specificity threshold. The pipeline was changed so that the
locus-neighbourhood search never populates the specificity field at all; only a genome-wide scan does, and
every other path fails safe to maximum risk.

The consequence is that the end-to-end run now **abstains**, returning `certified-negative` and zero
circuits. That is the correct outcome rather than a regression.

## 5. Demoted — leave-cell-line-out cross-validation

The leave-cell-line-out result, originally reported as showing roughly twice-lower residuals for real held-out
states than permuted nulls, leaked whole-panel statistics into each fold. It is retained as an internal
optimisation diagnostic and no longer reported as generalisation evidence.

## 6. Contextualised — subtype specificity and unsupervised recovery

Two further claims were reduced rather than retracted. Subtype-specific circuit design did not survive
validation: PANC-1 is not a clean model of either PDAC subtype, so subtype-conditioned results built on it
carry that caveat. And the observation that unsupervised methods surface GATA6 and KLF5 near the top is
positive-control recovery — these are known PDAC transcription factors — not discovery.

## What survived

<div class="callout pos">
<p>Of the project's substantive biological claims, <b>one</b> survived every attempt to kill it: prioritised
target promoters carry gained H3K27ac in PDAC relative to healthy pancreas, beyond expression, beyond the
selection variable, and beyond network hub-ness, holding across all twelve parameter settings tested.
It remains a single-cell-line result that ATAC does not replicate — see
<a href="{{ '/results/' | relative_url }}">Results</a> for the effect size and its limits.</p>
</div>

The four trained sequence models and the data-scaling analysis also stand, being straightforward held-out
measurements rather than mechanistic claims.
