---
layout: default
title: Results
subtitle: Every trained model, the data-scaling programme, and the one biological finding that survived.
permalink: /results/
---

## 1. Trained sequence models

Four learned models, each trained from scratch on real public data. Splits are leakage-controlled:
gene-grouped for guide RNAs, chromosome-held-out (test chr8 and chr9, validation chr7) for the
regulatory-part models. Ensemble weights are selected on the validation split and never on test.

<div class="tablewrap">
<table>
<thead><tr><th>Model</th><th>Module</th><th>Architecture</th><th>Metric</th><th class="num">Capped</th><th class="num">Full data</th><th class="num">Δ</th></tr></thead>
<tbody>
<tr><td><b>gRNA on-target</b></td><td>V</td><td>GBT + CNN</td><td>Spearman</td><td class="num">0.4938</td><td class="num up">0.6571</td><td class="num up">+0.1633</td></tr>
<tr><td><b>Promoter strength</b></td><td>II</td><td>GBT + CNN</td><td>Spearman</td><td class="num">0.5199</td><td class="num up">0.5275</td><td class="num up">+0.0075</td></tr>
<tr><td><b>Enhancer activity</b></td><td>II</td><td>multitask CNN</td><td>AUROC</td><td class="num">0.8087</td><td class="num up">0.8147</td><td class="num up">+0.0060</td></tr>
<tr><td><b>Promoter generator</b></td><td>VII</td><td>WGAN-GP</td><td>p90 strength</td><td class="num">0.937</td><td class="num up">0.992</td><td class="num up">+0.055</td></tr>
</tbody>
</table>
</div>

The generator is also scored on 4-mer Jensen–Shannon divergence against real promoters, where lower is
better: 0.0123 for the deployed model against 0.0508 for random DNA on the same reference, comfortably
inside the pre-registered 0.05 bound.

## 2. The data-scaling programme

Every model had been trained on an arbitrary fraction of its own available real data. The question in each
case was whether performance was limited by features and architecture, or simply by data volume.

### 2.1 gRNA on-target: adding a second real dataset

The shipped model was feature-saturated — Azimuth/Rule-Set-2 style additions bought +0.008 — but trained on
only 5,310 guides spanning **17 genes**. We added Kim et al. 2019, a real high-throughput SpCas9 library of
**12,832** synthetic-target guides in the identical 30-mer format, rank-normalised within dataset so it pools
with Doench's drug-gene rank.

The decisive check came *before* merging. A model trained on Doench's 17 genes and tested on all 12,832 Kim
guides scored Spearman **0.592** — higher than its own within-Doench held-out score of 0.53. That is only
possible if the model learned transferable guide biology rather than memorising 17 genes, which is what
justified combining the datasets at all.

<figure>
  <img src="{{ '/images/fig3_grna_components.png' | relative_url }}" alt="CNN, GBM and ensemble Spearman before and after adding the Kim-2019 dataset">
  <figcaption><b>Figure 4.</b> Component breakdown on the identical 688 held-out-gene guides. The CNN was the
  bottleneck: near-random at 0.392 on 17 genes, it reaches 0.617 on 18,142 guides. It had previously been
  down-weighted to 0.20 in the ensemble simply to stop it doing harm; the deployed ensemble is now a balanced
  0.40 CNN / 0.60 GBM.</figcaption>
</figure>

### 2.2 Promoter and enhancer: scaling curves

Rather than assert saturation from two points, each model was retrained at a ladder of training-set sizes and
scored at every point on the same fixed held-out test.

<figure>
  <img src="{{ '/images/fig2_scaling_curves.png' | relative_url }}" alt="Held-out performance versus training-set size for the promoter and enhancer models">
  <figcaption><b>Figure 5.</b> Performance versus training-set size on the fixed chr8/chr9 test. The promoter
  ensemble rises monotonically by +0.035 across an 18-fold increase in real data and then flattens over the
  final step — data-limited, then approaching saturation. The enhancer curve is flatter throughout, consistent
  with a model already near its ceiling. These are independent trainings, so run-to-run variation of roughly
  0.005 is expected; read the trend, not the third decimal.</figcaption>
</figure>

<div class="tablewrap">
<table>
<thead><tr><th>Promoter, n train</th><th class="num">10k</th><th class="num">20k</th><th class="num">40k</th><th class="num">80k</th><th class="num">120k</th><th class="num">181k</th></tr></thead>
<tbody>
<tr><td>CNN</td><td class="num">0.489</td><td class="num">0.477</td><td class="num">0.503</td><td class="num">0.514</td><td class="num">0.530</td><td class="num">0.528</td></tr>
<tr><td>Ensemble</td><td class="num">0.498</td><td class="num">0.501</td><td class="num">0.513</td><td class="num">0.519</td><td class="num">0.532</td><td class="num up">0.533</td></tr>
</tbody>
</table>
</div>

### 2.3 Where the data came from

<div class="tablewrap">
<table>
<thead><tr><th>Model</th><th>Cap</th><th>Real data available</th><th>Used after un-capping</th></tr></thead>
<tbody>
<tr><td>gRNA on-target</td><td>5,310 guides / 17 genes</td><td>+ 12,832 Kim-2019 guides</td><td class="num">18,142</td></tr>
<tr><td>Promoter strength</td><td>60,000 peaks</td><td>209,374 FANTOM5 CAGE peaks</td><td class="num">181,428</td></tr>
<tr><td>Enhancer activity</td><td>20,000 actives</td><td>470,874 ENCODE pancreas actives</td><td class="num">135,402</td></tr>
<tr><td>Promoter generator</td><td>12,000 promoters</td><td>52,342 top-quartile promoters</td><td class="num">52,342</td></tr>
</tbody>
</table>
</div>

The enhancer case is the starkest: the shipped model had been trained on roughly 4% of the accessible,
H3K27ac-marked pancreatic regions available to it.

## 3. Regulatory grammar transfers between healthy pancreas and PDAC

The enhancer model is trained on healthy pancreas. Testing it on PANC-1, a PDAC cell line, probes whether it
has learned tissue-specific artefacts or genuine regulatory grammar.

<figure>
  <img src="{{ '/images/fig6_cross_domain.png' | relative_url }}" alt="Enhancer AUROC within domain and across the healthy-pancreas to PDAC boundary in both directions">
  <figcaption><b>Figure 6.</b> A healthy-pancreas-only model predicts PANC-1 PDAC enhancers at AUROC 0.835 —
  <em>higher</em> than its own pancreas test — and the reverse direction holds at 0.790. The grammar is shared
  in both directions. The asymmetry is expected: the multi-donor pancreas set is the more diverse training
  source, so it generalises to a single cell line better than the reverse.</figcaption>
</figure>

This result is what explains a negative reported on the [validation page]({{ '/validation/' | relative_url }}):
because the grammar already transfers, merging PANC-1 data into training does not improve the pancreas
benchmark.

## 4. Promoter H3K27ac in PDAC — the surviving biological result

The 20 transcription factors surfaced by the targeting modules were derived **without using any PDAC
chromatin data**. Testing them against PDAC chromatin is therefore an out-of-sample check.

<div class="cards">
<div class="card"><div class="k up">+0.919</div><div class="l">mean log2 residual across 20 targets</div></div>
<div class="card"><div class="k">−0.091</div><div class="l">mean across 1,655 background loci</div></div>
<div class="card"><div class="k">0.0022</div><div class="l">Mann–Whitney p, one-sided</div></div>
<div class="card"><div class="k">70%</div><div class="l">of targets gain signal, versus 46% of background</div></div>
</div>

Measured on ENCODE fold-change-over-control bigWigs, PDAC (PANC-1) against healthy pancreas, over TSS ±2000 bp
on GRCh38. The strongest individual gains are HOXA3 (+5.88), FOSL1 (+2.77), MYBL2 (+2.19) and SMAD3 (+2.12);
GATA6 moves sharply the other way (−2.24), consistent with its known loss in the basal PDAC subtype.

<div class="callout">
<p><b>What limits this result.</b> It rests on a single PDAC cell line against one healthy fold-change track
per mark, where an earlier run averaged up to six — so the healthy reference is noisier and the comparison
correspondingly more conservative. The effect size is roughly 1.5–1.8 fold. ATAC accessibility does not
replicate it. It is a hypothesis worth testing in primary tumours, not an established fact.</p>
</div>

## 5. Circuit design and scoring

Modules III, IV and VI assemble the parts into candidate circuits and score them. Each circuit's Hill-ODE
parameters derive from its own components — transcription-factor expression, promoter × enhancer strength,
guide efficiency — rather than from a shared template. Robustness is a real per-circuit parameter sweep and
knockdown a real steady-state readout, and thousands of circuits are individually simulated rather than
scored by formula.

The end-to-end run currently returns **ABSTAIN** with certification `certified-negative`, emitting zero
circuits. This follows directly from repairing the off-target search: once specificity is evaluated
genome-wide instead of over a locus neighbourhood covering ~0.001% of the genome, no candidate guide clears
the pre-registered threshold. Reporting zero circuits is the correct result, because absence of a search is
not evidence of specificity.
