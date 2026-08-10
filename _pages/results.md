---
layout: default
title: Results
subtitle: The four trained models, the data-scaling programme behind them, and the one biological finding that held.
permalink: /results/
---

## Trained sequence models

Four learned models sit inside the pipeline, and all four were trained from scratch on real public data.
Splits are chosen so that the held-out set cannot share structure with the training set. Guide models use a
gene-grouped split, implemented in `src/pdac_circuit/grna/training.py`, so that no gene contributes guides to
both sides and a model cannot succeed by memorising the local sequence neighbourhood of a gene it has already
seen. The regulatory-part models use a chromosome-held-out split from
`src/pdac_circuit/harness/splits.py`, holding chr8 and chr9 for test and chr7 for validation. Ensemble
weights are selected on the validation split and never on test.

<div class="tablewrap">
<table>
<thead><tr><th>Model</th><th>Module</th><th>Architecture</th><th>Metric</th><th class="num">Capped</th><th class="num">Full data</th><th class="num">Δ</th></tr></thead>
<tbody>
<tr><td>gRNA on-target</td><td>V</td><td>GBT + CNN</td><td>Spearman</td><td class="num">0.4938</td><td class="num up">0.6571</td><td class="num up">+0.1633</td></tr>
<tr><td>Promoter strength</td><td>II</td><td>GBT + CNN</td><td>Spearman</td><td class="num">0.5199</td><td class="num up">0.5275</td><td class="num up">+0.0075</td></tr>
<tr><td>Enhancer activity</td><td>II</td><td>multitask CNN</td><td>AUROC</td><td class="num">0.8300</td><td class="num up">0.8375</td><td class="num">+0.0074</td></tr>
<tr><td>Promoter generator</td><td>VII</td><td>WGAN-GP</td><td>p90 strength</td><td class="num">0.937</td><td class="num up">0.992</td><td class="num up">+0.055</td></tr>
</tbody>
</table>
</div>

Two of these deltas are small enough to need a caveat stated with them. An independent retraining of the
same model on the same data varies by roughly 0.005, so the promoter and enhancer gains sit inside the
variation of the procedure that produced them and are not established by the before-and-after comparison on
its own. The evidence for the promoter is the scaling curve in the next section, whose six points increase
without a single reversal. The enhancer has no equivalent support, and the attempt to give it some by
supplying far more real data is reported in its own section below. The gRNA gain is more than thirty times
that variation and needs no such qualification.

The enhancer figures in the table are measured on the enriched dataset described later, where the previously
deployed model was re-scored on the same test so that the two are comparable. They are not comparable with
the 0.8147 that appeared here previously, because that was measured on a smaller test drawn from a smaller
peak set.

The generator carries a second metric because realism and usefulness are separate properties for a
generative model. On 4-mer Jensen-Shannon divergence against real promoters, where a lower value indicates
closer composition, the deployed generator scores 0.0123 while random DNA scores 0.0508 against the same
reference. Both sit comfortably inside the pre-registered bound of 0.05.

## The data-scaling programme

Each model was tested for whether its performance was bounded by features and architecture or simply by the
amount of data it had been allowed to see. The question is worth asking carefully, because the two failure
modes call for opposite responses, and the intuition that a model has "converged" is unreliable when the
training set was capped for reasons that had nothing to do with the science.

### gRNA on-target, where a second real dataset was added

The shipped model was close to saturated with respect to features. Adding Azimuth and Rule-Set-2 style
descriptors through `scripts/grna_feature_upgrade.py` bought roughly 0.008 Spearman, which is not the profile
of a model starved of signal in its inputs. It was, however, badly starved of data, having been fitted on
5,310 guides spanning 17 genes.

Kim et al. 2019 supplies 12,832 high-throughput SpCas9 guides in the identical 30-mer context format, and it
was added through `load_kim2019` in `src/pdac_circuit/grna/datamodule.py`. Because the two datasets measure
different endpoints, Doench reporting a drug-gene rank and Kim reporting background-subtracted indel
frequency, the Kim endpoint is rank-normalised within its own dataset before pooling. Spearman is the
reporting metric, so a monotone within-dataset transform is the appropriate way to combine them without
asserting that a rank in one assay equals a percentage in the other.

The decision to pool at all rested on a test run beforehand rather than afterwards. A model trained only on
Doench's 17 genes and evaluated on all 12,832 Kim guides reached Spearman 0.592, which exceeds its own
within-Doench held-out score of 0.53. A model that had merely memorised 17 genes could not have done that.
Had the number come back near chance, the two datasets would not have been combined.

<figure>
  <img src="{{ '/images/fig3_grna_components.png' | relative_url }}" alt="CNN, GBM and ensemble Spearman before and after adding the Kim-2019 dataset">
  <figcaption><b>Figure 4.</b> Component breakdown on the identical 688 held-out-gene guides. The CNN was the
  binding constraint. At 0.392 on 17 genes it was close to useless, and it had been down-weighted to 0.20 in
  the ensemble largely to stop it doing harm. On 18,142 guides it reaches 0.617, and the deployed ensemble is
  a balanced 0.40 CNN against 0.60 GBM. Generated by <code>scripts/grna_cnn_kim_retrain.py</code>.</figcaption>
</figure>

### The enhancer, given every peak file ENCODE has

The enhancer was the weakest quantitative claim on this site, so it was given the most direct remedy
available, which is more real data. Two separate limits were found. The peak loader in
`src/pdac_circuit/data/tracks.py` carried `max_files: int = 4`, so the project had only ever retrieved four
ATAC and four H3K27ac pancreas peak files when ENCODE has released ten and fourteen. Fetching the remaining
sixteen through `scripts/fetch_all_pancreas_peaks.py` raised the ATAC interval count from 874,795 to
1,974,976 and the accessible, H3K27ac-marked regions from 470,874 to 1,376,493.

<div class="callout neg">
<p>An audit run afterwards showed that those interval counts are not counts of distinct regions, and an
earlier version of this page described them as though they were. Peaks are pooled across every released
experiment without merging, so a region called in several experiments is counted once per experiment. Of
the 1,974,976 ATAC intervals only 684,673 are distinct coordinates, a duplication factor of 2.88 and a
redundant fraction of 65.3 per cent. For H3K27ac the factor is 1.36. Because
<code>scripts/enhancer_maxdata.py</code> iterates the pooled intervals without deduplication, a region
observed in several experiments contributes one training row per observation, so the training set grew by
considerably less in distinct regions than the row count suggests. The numbers come from
<code>scripts/peak_duplication_audit.py</code> and are recorded in
<code>results/peak_duplication_audit.json</code>.</p>
</div>

The duplication does not put the held-out comparison at risk, because the split is by chromosome and every
copy of a duplicated region falls on the same side of it. What it changes is the interpretation. The
additional files do carry real replication across donors and experiments, and repeated observation of the
same region is evidence about that region rather than noise, but it is not the same thing as covering more
of the genome.

The second limit was that the shared trainer moves its whole tensor onto the GPU, and one-hot sequence at
2,000 bp in float32 costs 32 kB per row, so the previous 135,402-row set already occupied 4.3 GB of a 12 GB
card. Training on the enriched data required storing base indices as `int8` on the host and building the
one-hot per batch on the device, which is sixteen times smaller and is implemented in
`scripts/enhancer_maxdata.py`.

The model was then retrained on 540,199 rows, eight times the previous training set in rows though by less
than that in distinct regions for the reason given above, and evaluated on all
54,103 chr8 and chr9 rows of the enriched dataset. The previously deployed model was re-scored on that same
test, giving 0.8300 against 0.8375 for the new one.

<div class="callout neg">
<p>The gain of +0.0074 is roughly 1.2 times the 0.0063 non-monotonicity of the curve itself, so it does not
clear the two-fold bar applied to every other claim here and is not established as an improvement. The curve
is still not monotone, falling from 0.8334 to 0.8272 when the training set doubled from 135,049 to 270,099
rows. The signal-regression head is marginally worse at 0.5162 against 0.5213. Eight times the data, drawn
from three times the biological samples, did not move this benchmark in a way that survives its own noise.
The reasonable conclusion is that the enhancer classifier is near its ceiling for this architecture and that
data volume is not the binding constraint. The retrained model is kept because it rests on far more
replication, not because the benchmark moved.</p>
</div>

### Promoter and enhancer, measured as curves rather than as two points

Declaring a model saturated on the basis of a before-and-after pair is weak evidence, since a single
comparison cannot distinguish a model at its ceiling from a model that happened to gain little on that
particular step. Both models were therefore retrained at a ladder of training-set sizes by
`scripts/promoter_scaling_curve.py` and `scripts/enhancer_scaling_curve.py`, with every point scored on the
same fixed held-out test.

<figure>
  <img src="{{ '/images/fig2_scaling_curves.png' | relative_url }}" alt="Held-out performance versus training-set size for the promoter and enhancer models">
  <figcaption><b>Figure 5.</b> Performance against training-set size on the fixed chr8 and chr9 test. The
  promoter ensemble climbs at every one of six sizes without a reversal, by 0.035 in total across an
  eighteen-fold increase in data, then flattens over the final step. A strictly increasing run of six points
  has probability 1/6! = 0.0014 under a null of no trend, which is what makes the promoter result credible
  where its single delta does not. The enhancer curve spans only 0.007 in total and is not monotone. These are independent trainings, so run-to-run
  variation of roughly 0.005 is expected and the trend rather than any single point is the result.</figcaption>
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

The division of labour inside the promoter ensemble is informative. The convolutional network carries
essentially all of the gain, moving from 0.489 to 0.528, while the tree model over 4-mer features stays flat
across the whole range. That is the expected pattern if the additional data is supplying sequence context
that a k-mer summary cannot represent, and it is why the validation-selected ensemble weight shifts steadily
toward the network as the training set grows.

### What was available against what was used

<div class="tablewrap">
<table>
<thead><tr><th>Model</th><th>Cap</th><th>Real data available</th><th class="num">Used after un-capping</th></tr></thead>
<tbody>
<tr><td>gRNA on-target</td><td>5,310 guides across 17 genes</td><td>plus 12,832 Kim-2019 guides</td><td class="num">18,142</td></tr>
<tr><td>Promoter strength</td><td>60,000 peaks</td><td>209,374 FANTOM5 CAGE peaks</td><td class="num">181,428</td></tr>
<tr><td>Enhancer activity</td><td>20,000 actives, 8 peak files</td><td>1,376,493 actives across 24 peak files</td><td class="num">540,199</td></tr>
<tr><td>Promoter generator</td><td>12,000 promoters</td><td>52,342 top-quartile promoters</td><td class="num">52,342</td></tr>
</tbody>
</table>
</div>

## Regulatory grammar transfers between healthy pancreas and PDAC

The enhancer model is trained entirely on healthy pancreatic chromatin, which raises an obvious question
about whether it has learned general regulatory grammar or the idiosyncrasies of one tissue preparation.
Testing it on PANC-1, a PDAC cell line it has never seen, separates those possibilities.

<figure>
  <img src="{{ '/images/fig6_cross_domain.png' | relative_url }}" alt="Enhancer AUROC within domain and across the healthy-pancreas to PDAC boundary in both directions">
  <figcaption><b>Figure 6.</b> A healthy-pancreas model predicts PANC-1 PDAC enhancers at AUROC 0.835, which
  is higher than its score on its own domain, and the reverse direction holds at 0.790. The asymmetry follows
  from the composition of the two training sets, since the multi-donor pancreas panel is the more diverse
  source and generalises to a single cell line more readily than one cell line generalises to a panel.
  Computed by <code>scripts/enhancer_panc1_augment.py</code> and
  <code>scripts/enhancer_scaling_curve.py</code>.</figcaption>
</figure>

This result is what explains a negative reported on the [validation page]({{ '/validation/' | relative_url }}).
Because the grammar already transfers in both directions, adding PANC-1 data to training cannot supply
information the model was missing, and in practice it slightly degrades the pancreatic benchmark.

## Promoter H3K27ac in PDAC

The 20 transcription factors examined here were surfaced by the targeting and attractor modules without any
PDAC chromatin entering the selection. Testing them against PDAC chromatin is therefore a genuine
out-of-sample comparison rather than a circular restatement of how they were chosen.

<div class="cards">
<div class="card"><div class="k up">+0.919</div><div class="l">mean log2 residual across 20 targets</div></div>
<div class="card"><div class="k">−0.091</div><div class="l">mean across 1,655 background loci</div></div>
<div class="card"><div class="k">0.0022</div><div class="l">Mann-Whitney p, one-sided</div></div>
<div class="card"><div class="k">70%</div><div class="l">of targets gain signal, against 46% of background</div></div>
</div>

<figure>
  <img src="{{ '/images/fig7_evidence_heatmap.png' | relative_url }}" alt="Heatmap of multi-omic evidence layers across the prioritised target genes">
  <figcaption><b>Figure 7.</b> Multi-omic evidence for each prioritised target, with every layer standardised
  within its own column so that quantities on different scales can be read side by side. Red indicates a
  higher value within a layer and blue a lower one. Hatched cells are layers in which that gene was not
  measured, and two columns are hatched throughout, which records that promoter methylation and the ATAC
  residual are unavailable for this target set rather than that they were zero. The pattern is deliberately
  uneven, since no single layer nominates a target and the prioritisation depends on agreement across
  several. Assembled by <code>scripts/rac_target_dossiers.py</code>.</figcaption>
</figure>

Signal is measured on ENCODE fold-change-over-control bigWigs comparing PANC-1 against healthy pancreas over
TSS ±2000 bp on GRCh38, with the comparison implemented in `scripts/pdac_residual_foldchange.py`. The
strongest individual gains are HOXA3 at +5.88, FOSL1 at +2.77, MYBL2 at +2.19 and SMAD3 at +2.12. GATA6 moves
sharply in the opposite direction at −2.24, which is consistent with its established loss in the basal
subtype and is a reassuring sign that the measurement is tracking real biology rather than a processing
artefact.

The result was stress-tested rather than reported once. `scripts/h3k27ac_fragility.py` varied the analysis
parameters, `scripts/h3k27ac_pseudocount.py` varied the pseudocount used in the log ratio, and
`scripts/h3k27ac_window_and_loci.py` varied the promoter window and the background locus definition. The
contrast held across all twelve settings tested.

<div class="callout">
<p>The limits are real and worth stating plainly. The comparison rests on a single PDAC cell line against one
healthy fold-change track per mark, whereas an earlier run averaged up to six healthy tracks, so the healthy
reference here is noisier and the test is correspondingly more conservative. The effect size is roughly 1.5
to 1.8 fold, which is modest. ATAC accessibility does not replicate it, and that non-replication is reported
in the <a href="{{ '/addenda/chromatin/' | relative_url }}">chromatin addendum</a> rather than omitted. This
is a hypothesis that deserves testing in primary tumours, not an established fact.</p>
</div>

## Circuit assembly and scoring

Modules III, IV and VI assemble parts into candidate circuits and score them. Each circuit's Hill-ODE
parameters are derived from its own components, meaning the transcription factor's expression, the product of
its promoter and enhancer strengths, and its guide efficiency, rather than from a shared template applied
across every design. Robustness is computed as an actual per-circuit parameter sweep in
`src/pdac_circuit/circuit/stability.py`, and knockdown is read from a steady-state solution rather than
assumed. Several thousand circuits are individually simulated in the deep design path rather than ranked by a
closed-form score.

<figure>
  <img src="{{ '/images/fig9_circuit.png' | relative_url }}" alt="Schematic of a designed CRISPRi circuit targeting GATA6 with negative feedback">
  <figcaption><b>Figure 8.</b> One designed circuit, drawn from the classical-subtype deep run. A synthetic
  promoter taken from the generated library, together with an enhancer scored by the activity model, drives a
  dCas9-KRAB effector, which is directed to the GATA6 locus by a guide scored by the on-target model. Blunt
  heads denote repression and the dashed edge is the negative feedback that makes the steady state
  self-limiting. Kinetics follow the Hill form shown, with the promoter supplying the production rate
  and each incoming edge contributing an activating or repressive term.</figcaption>
</figure>

The end-to-end run currently returns ABSTAIN with certification `certified-negative` and emits zero circuits.
This follows directly from repairing the off-target search. Once specificity is evaluated genome-wide by
`scripts/genomewide_offtarget_audit.py` instead of over a locus neighbourhood, no candidate guide clears the
pre-registered threshold, and `scripts/offtarget_cutoff_sensitivity.py` confirms that the conclusion is not an
artefact of where the cutoff was placed. Reporting zero circuits is the correct outcome here, because the
absence of a genome-wide search is not evidence of specificity.
