---
layout: default
title: Overview
---

<div class="hero">
<h1>Designing synthetic gene circuits for pancreatic cancer, and testing every claim until it breaks</h1>
<p class="lede">A from-scratch computational pipeline that ranks pancreatic ductal adenocarcinoma
transcription-factor targets, designs regulatory parts and CRISPRi guides against them, and simulates the
resulting circuits. Everything is trained on real public genomics data. Roughly half of the total effort went
into trying to disprove the pipeline's own results, and most of the original claims did not survive that.</p>
</div>

The pipeline is organised as eight modules spanning 17 Python packages and about 130 modules under
`src/pdac_circuit`, with 51 analysis entrypoints under `scripts/`. Nothing in it depends on an external
pretrained model for features, pseudo-labels, or training targets. Frozen third-party predictors are allowed
in one place only, as hash-locked evaluation baselines, and never as inputs to a design decision. Four of the
eight modules contain learned sequence models that were trained from scratch and evaluated on
leakage-controlled held-out splits.

What follows reports three things. The first is what those four models actually achieve once they are given
the full quantity of real data available to them, which turned out to matter far more than any architectural
change. The second is the single biological finding that survived every attempt made to kill it. The third,
given the same prominence as the other two, is the set of claims that were withdrawn.

## Headline results

<div class="cards">
<div class="card"><div class="k">0.657</div><div class="l">gRNA on-target Spearman on held-out genes, raised from 0.494</div></div>
<div class="card"><div class="k">0.815</div><div class="l">enhancer-activity AUROC on held-out chromosomes</div></div>
<div class="card"><div class="k">+0.92</div><div class="l">log2 H3K27ac gain at target promoters in PDAC, p = 0.0022</div></div>
<div class="card"><div class="k">0.55<span style="font-size:1rem;color:var(--mut)"> vs </span>0.63</div><div class="l">the original attractor claim, beaten by plain network degree</div></div>
</div>

### Every learned model was limited by data rather than by architecture

Each of the four trained models had been fitted on an arbitrary fraction of the real data that was already
sitting on disk for it. The gRNA model saw 5,310 guides drawn from only 17 genes. The promoter model saw
60,000 of 209,374 FANTOM5 CAGE peaks. The enhancer model saw 20,000 of 470,874 accessible, H3K27ac-marked
pancreatic regions, which is roughly four percent of what was available. The generator saw 12,000 of 52,342
high-activity promoters. Removing those caps produced the largest single block of improvement in the project,
and it did so without changing a single layer of any network.

<figure>
  <img src="{{ '/images/fig1_scaleup.png' | relative_url }}" alt="Held-out performance before and after removing training-data caps, for four models">
  <figcaption><b>Figure 1.</b> Held-out performance before and after the caps were removed. Each comparison is
  like for like, because the previously deployed model is re-scored on the identical held-out set rather than
  compared against its historical reported figure. Grey hatched bars are the capped models and blue bars are
  the full-data models. The y-scales are independent, so note that the gRNA gain is an order of magnitude
  larger than the other three. Produced by <code>scripts/promoter_scaleup.py</code>,
  <code>scripts/enhancer_scaleup.py</code>, <code>scripts/grna_cnn_kim_retrain.py</code> and
  <code>scripts/promoter_gan_scaleup.py</code>.</figcaption>
</figure>

### One biological finding survived adversarial validation

The 20 transcription factors that the targeting modules surfaced sit on promoters which gain H3K27ac in PDAC
relative to healthy pancreas. That contrast holds against 1,655 matched background loci, and it was
established without any PDAC chromatin entering the target-selection process, so it is an out-of-sample test
rather than a restatement of the selection criterion.

<figure>
  <img src="{{ '/images/fig4_h3k27ac.png' | relative_url }}" alt="Per-target log2 H3K27ac fold-change residual between PDAC and healthy pancreas">
  <figcaption><b>Figure 2.</b> Promoter H3K27ac in PDAC against healthy pancreas for the 20 prioritised
  targets. The target mean sits at +0.919 log2 while the background mean sits at −0.091, and 70 percent of
  targets gain signal against 46 percent of background. Measured on ENCODE fold-change-over-control tracks by
  <code>scripts/pdac_residual_foldchange.py</code>, with the parameter sweeps in
  <code>scripts/h3k27ac_fragility.py</code>, <code>scripts/h3k27ac_pseudocount.py</code> and
  <code>scripts/h3k27ac_window_and_loci.py</code>.</figcaption>
</figure>

### The original headline claim was withdrawn

The project's novel contribution was a bistable attractor model whose collapse score was reported to predict
CRISPR essentiality and to beat network centrality. The permutation test behind that claim compared the score
against chance, which it beat, and never against the baseline it claimed to outperform. When
`scripts/rigorous_validation.py` ran the head-to-head comparison directly, the result reversed.

<figure>
  <img src="{{ '/images/fig5_rac_validation.png' | relative_url }}" alt="Attractor collapse versus network degree as predictors of essentiality, with confidence interval on the difference">
  <figcaption><b>Figure 3.</b> Attractor collapse reaches AUC 0.547 against plain network degree at 0.629,
  measured across 419 genes carrying 31 essential positives. The paired-bootstrap confidence interval on the
  difference spans zero, and once degree is controlled for, the score contributes nothing further. The claim
  is retracted.</figcaption>
</figure>

## What this is, and what it is not

This is a computational prototype together with an auditing framework built around it. It has produced no
wet-lab result, no cloning-ready construct, and no experimentally validated guide RNA. After the off-target
search in `src/pdac_circuit/grna/genome_offtarget.py` was repaired to scan the whole genome instead of a
window around each target locus, no candidate guide cleared the pre-registered specificity threshold, and the
end-to-end run now returns a certified negative with zero circuits. That behaviour is intended. A search that
covers roughly a thousandth of a percent of the genome cannot establish specificity, so reporting its silence
as safety would have been the error.

The contributions that do stand are the multi-omic data assembly described on the
[data page]({{ '/data/' | relative_url }}), the four trained sequence models and the scaling analysis behind
them, and a body of methodology for not fooling yourself that is documented in full across the
[review arc]({{ '/reports/review/' | relative_url }}).

## Read further

The summary pages are backed by the project's primary documents, all reproduced here in full. Four
[addenda]({{ '/addenda/' | relative_url }}) cover individual analyses at depth, including the dynamics audit
which found that the fitted attractor system is not bistable at any gain. The
[written record]({{ '/reports/' | relative_url }}) contains the compendium, the technical report, the complete
methods, and the twenty-eight step review arc in which most of the original claims were tested and overturned.
Exact held-out numbers for every model, split and comparison are tabulated on the
[evaluation page]({{ '/evaluation/' | relative_url }}).

<div class="next">
<a href="{{ '/results/' | relative_url }}">Results →</a>
<a href="{{ '/evaluation/' | relative_url }}">Evaluation →</a>
<a href="{{ '/methods/' | relative_url }}">Methods →</a>
<a href="{{ '/validation/' | relative_url }}">Validation →</a>
<a href="{{ '/data/' | relative_url }}">Data →</a>
<a href="{{ '/addenda/' | relative_url }}">Addenda →</a>
<a href="{{ '/reports/' | relative_url }}">Full reports →</a>
</div>
