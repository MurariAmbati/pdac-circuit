---
layout: default
title: Overview
---

<div class="hero">
<h1>Designing synthetic gene circuits for pancreatic cancer, and testing every claim until it breaks</h1>
<p class="lede">A from-scratch computational pipeline that ranks pancreatic-ductal-adenocarcinoma (PDAC)
transcription-factor targets, designs regulatory parts and CRISPRi guides against them, and simulates the
resulting circuits — trained entirely on real public genomics data. The second half of the work was
spent trying to disprove its own results.</p>
</div>

The pipeline is eight modules, all built from scratch: no external pretrained model supplies features,
pseudo-labels, or training targets. Four of those modules contain learned sequence models trained on real
data with leakage-controlled held-out splits. This site reports what those models achieve, how the circuits
are assembled and scored, and — with equal prominence — which claims did not survive validation.

## Headline results

<div class="cards">
<div class="card"><div class="k">0.657</div><div class="l">gRNA on-target Spearman, held-out genes, up from 0.494</div></div>
<div class="card"><div class="k">0.815</div><div class="l">enhancer-activity AUROC, held-out chromosomes</div></div>
<div class="card"><div class="k">+0.92</div><div class="l">log2 H3K27ac gain at target promoters, PDAC vs healthy (p = 0.0022)</div></div>
<div class="card"><div class="k">0.55<span style="font-size:1rem;color:var(--mut)"> vs </span>0.63</div><div class="l">the headline attractor claim lost to plain network degree</div></div>
</div>

**Four learned models, all limited by data rather than architecture.** Every trained model in the pipeline
turned out to have been capped at an arbitrary fraction of the real data available to it. Removing those caps
is the largest single source of improvement in the project.

<figure>
  <img src="{{ '/images/fig1_scaleup.png' | relative_url }}" alt="Held-out performance before and after removing training-data caps, for four models">
  <figcaption><b>Figure 1.</b> Held-out performance before and after removing training-data caps. Each
  comparison is apples-to-apples: the previously shipped model is re-scored on the identical held-out set.
  Grey hatched bars are the capped models, blue are the full-data models. Note the independent y-scales —
  the gRNA gain is an order of magnitude larger than the others.</figcaption>
</figure>

**One biological finding survived adversarial validation.** The transcription factors the pipeline
prioritises sit on promoters that gain H3K27ac in PDAC relative to healthy pancreas, and the effect holds
against matched background loci.

<figure>
  <img src="{{ '/images/fig4_h3k27ac.png' | relative_url }}" alt="Per-target log2 H3K27ac fold-change residual between PDAC and healthy pancreas">
  <figcaption><b>Figure 2.</b> Promoter H3K27ac in PDAC versus healthy pancreas for the 20 prioritised
  targets. Target mean +0.919 log2 against a background mean of −0.091 across 1,655 loci; 70% of targets gain
  signal versus 46% of background. Measured on ENCODE fold-change-over-control tracks.</figcaption>
</figure>

**The original headline claim was retracted.** The project's novel contribution was a bistable
attractor model whose "collapse" score was claimed to predict CRISPR essentiality and beat network
centrality. Given a head-to-head test it had never been given, it did not.

<figure>
  <img src="{{ '/images/fig5_rac_validation.png' | relative_url }}" alt="Attractor collapse versus network degree as predictors of essentiality, with confidence interval on the difference">
  <figcaption><b>Figure 3.</b> Attractor collapse (0.547) loses to plain network degree (0.629) at predicting
  held-out essentiality across 419 genes. The paired-bootstrap confidence interval on the difference spans
  zero, and controlling for degree the score contributes nothing. Retracted.</figcaption>
</figure>

## What this is, and what it is not

This is a computational prototype and an auditing framework. It has produced no wet-lab result, no
cloning-ready construct, and no experimentally validated guide RNA. After the off-target search was repaired
to cover the whole genome rather than a locus neighbourhood, the end-to-end circuit run **abstains** rather
than emit designs resting on numbers it cannot support — which is the intended behaviour.

The durable contributions are the multi-omic data assembly, the four trained sequence models, the
data-scaling analysis, and a body of methodology for not fooling yourself.

<div class="next">
<a href="{{ '/results/' | relative_url }}">Results →</a>
<a href="{{ '/methods/' | relative_url }}">Methods →</a>
<a href="{{ '/validation/' | relative_url }}">Validation →</a>
<a href="{{ '/data/' | relative_url }}">Data →</a>
</div>
