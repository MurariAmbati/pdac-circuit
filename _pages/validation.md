---
layout: default
title: Validation
subtitle: The claims that did not survive, reported with the same prominence as the ones that did.
permalink: /validation/
---

Most of the value in this project sits on this page. A pipeline that reports only its successes is not
measuring anything, and the tests below were built specifically to break the project's own results. Several
of them did. Each analysis has a named entrypoint under `scripts/`, so the test that overturned a claim can
be run again by anyone who wants to check it.

## Retracted, the claim that attractor collapse predicts essentiality

Module VIII modelled the PDAC regulatory network as a bistable dynamical system of the form
`x ← σ(gain · (Wx + b))`, with the weight matrix masked to a DepMap co-expression graph and sign-anchored to
the observed correlations. The 54 real PDAC cell states were fitted as the viable attractor while a low state
was penalised to remain dead. Clamping a gene downward and measuring how far the whole network collapsed
toward the dead attractor produced a collapse score, and that score was reported to predict held-out CRISPR
essentiality at AUC 0.653 with a 50,000-permutation p of 0.0022, beating network degree at 0.629 and
eigenvector centrality at 0.584.

The flaw was in what the permutation test compared against. Shuffling the labels tests a score against
chance, and the collapse score did beat chance. It never tested the score against the baseline it claimed to
outperform, which is a different and much harder comparison. `scripts/rigorous_validation.py` ran that
comparison directly, using a paired bootstrap over the same genes so that the two predictors are assessed on
identical data rather than on independently drawn samples.

<figure>
  <img src="{{ '/images/fig5_rac_validation.png' | relative_url }}" alt="Attractor collapse versus degree and eigenvector centrality, with a confidence interval on the difference">
  <figcaption><b>Figure 7.</b> Across 419 genes carrying 31 essential positives, attractor collapse reaches
  AUC 0.547 while plain network degree reaches 0.629. The paired-bootstrap 95 percent confidence interval on
  the difference spans zero.</figcaption>
</figure>

<div class="tablewrap">
<table>
<thead><tr><th>Statistic</th><th class="num">Value</th></tr></thead>
<tbody>
<tr><td>AUC, attractor collapse</td><td class="num dn">0.5471</td></tr>
<tr><td>AUC, network degree</td><td class="num up">0.6290</td></tr>
<tr><td>AUC, eigenvector centrality</td><td class="num">0.5843</td></tr>
<tr><td>Δ AUC, collapse minus degree</td><td class="num dn">−0.0819</td></tr>
<tr><td>95 percent CI, paired bootstrap</td><td class="num">[−0.199, +0.029]</td></tr>
<tr><td>p, two-sided</td><td class="num">0.147</td></tr>
<tr><td>PR-AUC, collapse against degree</td><td class="num">0.098 against 0.131</td></tr>
<tr><td>Partial Spearman given degree, expression, variance</td><td class="num">0.028, p = 0.563</td></tr>
<tr><td>Cross-validated AUC, covariates only then plus collapse</td><td class="num">0.653 then 0.652</td></tr>
</tbody>
</table>
</div>

The conclusion is that collapse adds nothing beyond network degree. Once degree, expression and variance are
controlled for, its partial correlation with essentiality is 0.028 at p of 0.56, and adding it to a
covariates-only model moves cross-validated AUC by −0.001. The original 0.653 arose because the model
configuration had been selected using the same CRISPR labels that were then used to report the result, which
is a form of selection that a permutation against chance cannot detect.

A separate audit in `scripts/dynamics_characterization.py` and `scripts/verify_dynamics_instability.py` went
further and examined whether the fitted system is bistable at all. It is not, at any gain tested, which means
the mechanism the claim rested on does not exist in the form described. `scripts/gain_sweep_rescue.py` was
written specifically to look for a gain regime in which bistability might be recovered, and it did not find
one. The formulation survives only as a way of expressing interventions, not as a predictor. The full
treatment is in the [dynamics addendum]({{ '/addenda/dynamics/' | relative_url }}), and the rebuilt second
version of the model is documented in the [v2 addendum]({{ '/addenda/rac-v2/' | relative_url }}).

## Negative, adding PDAC chromatin to the enhancer model

A genuinely new real dataset does not automatically improve a model, and treating acquisition as if it were
improvement is an easy mistake to make. Adding PANC-1 PDAC ATAC and H3K27ac to the healthy-pancreas enhancer
training set was therefore tested under exactly the same controlled protocol as every other data addition, in
`scripts/enhancer_panc1_augment.py`.

<div class="tablewrap">
<table>
<thead><tr><th>Training set</th><th class="num">Pancreas test</th><th class="num">PANC-1 test</th></tr></thead>
<tbody>
<tr><td>Pancreas only</td><td class="num up">0.8150</td><td class="num">0.8349</td></tr>
<tr><td>Pancreas plus PANC-1</td><td class="num dn">0.8096</td><td class="num">0.8306</td></tr>
</tbody>
</table>
</div>

Merging costs 0.0054 AUROC on the benchmark, so the merged model was not deployed. The explanation is visible
in the same table, because the pancreas-only model already scores 0.835 on PANC-1, which is better than it
scores on its own domain. The model was never short of enhancer sequence grammar. It was short of pancreatic
sample count, and mixing in a second domain dilutes rather than supplements. This negative is worth recording
precisely because the cross-domain measurement explains why it happened.

## Negative, un-capping the generator did not improve realism

Removing the generator's 12,000-promoter cap and retraining on all 52,342 sequences did not lower 4-mer
divergence. It moved from 0.0088 to 0.0123, slightly worse, and within the run-to-run variation expected of a
GAN whose divergence already sat close to its floor.

It did substantially strengthen the selectable tail, moving 90th-percentile predicted strength from 0.937 to
0.992 and median uplift from −0.021 to +0.106. That is the axis the pipeline actually consumes, since it
selects the strongest promoter from a generated library rather than a random member of it, and the un-capped
generator is deployed on that basis with the realism cost stated rather than hidden. Both versions clear the
pre-registered certification.

One point is worth making explicit, because a stronger score on a model-derived metric usually invites the
suspicion that the generator has learned to game the scorer. It has not, and it cannot have, because the GAN
trains only against real sequence through its critic and never sees the promoter model during training. The
stronger tail is therefore a property of the generated distribution rather than an artefact of optimising
against the evaluator. The memorisation check in `scripts/gan_memorisation_test.py` separately confirms the
generator is not reproducing training sequences.

## Retracted, an off-target risk of 0.00

Guides were originally reported with an off-target risk of 0.00. That number came from searching a window
around the target locus covering roughly a thousandth of a percent of the genome, which is not a specificity
measurement in any meaningful sense.

Rescored with the exact Doench-2016 CFD nucleotide-pair matrix over a genome-wide scan in
`scripts/genomewide_offtarget_audit.py`, all four final guides fail the pre-registered specificity threshold.
`scripts/offtarget_cutoff_sensitivity.py` confirms that this conclusion does not depend on where the cutoff
was placed. The code was then changed so that the locus-neighbourhood search never populates the specificity
field at all, with only a genome-wide scan able to do so and every other path failing safe to maximum risk.

The consequence is that the end-to-end run now abstains, returning a certified negative with zero circuits.
That is the correct outcome rather than a regression, and it is the clearest single illustration of the
principle that absence of a search is not evidence of specificity.

## Demoted, leave-cell-line-out cross-validation

The leave-cell-line-out result was originally reported as showing roughly twice-lower residuals for real
held-out states than for permuted nulls. It leaked whole-panel statistics into each fold, since quantities
computed across the full panel were available to every fold rather than recomputed within it. It is retained
as an internal optimisation diagnostic and is no longer reported as evidence of generalisation.

## Contextualised, subtype specificity and unsupervised recovery

Two further claims were reduced rather than withdrawn. Subtype-specific circuit design did not survive
validation, because PANC-1 is not a clean model of either PDAC subtype, so any subtype-conditioned result
resting on it inherits that ambiguity. The relevant analysis is in `scripts/subtype_resolved_targets.py`.
Separately, the observation that unsupervised methods surface GATA6 and KLF5 near the top of the ranking is
positive-control recovery rather than discovery, since both are well established PDAC transcription factors
and recovering them indicates the method is working rather than that something new has been found.

## What survived

<div class="callout pos">
<p>Of the project's substantive biological claims, one survived every attempt to kill it. Prioritised target
promoters carry gained H3K27ac in PDAC relative to healthy pancreas, and the contrast holds beyond expression,
beyond the variable used to select the targets, and beyond network hub-ness, across all twelve parameter
settings tested. It remains a single-cell-line result that ATAC does not replicate, and the effect size is
modest at roughly 1.5 to 1.8 fold. The full accounting is in the
<a href="{{ '/results/' | relative_url }}">results</a> and the
<a href="{{ '/addenda/chromatin/' | relative_url }}">chromatin addendum</a>.</p>
</div>

The four trained sequence models and the data-scaling analysis also stand, though it is worth being clear
about why they are on firmer ground. They are straightforward held-out measurements of predictive performance
rather than mechanistic claims about biology, so the ways in which they could be wrong are narrower and
easier to close off.
