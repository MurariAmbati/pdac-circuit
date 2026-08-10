---
layout: default
title: Methods
subtitle: The eight modules, how each model is trained and evaluated, and the rules applied to every number.
permalink: /methods/
---

## Pipeline architecture

The pipeline is eight modules implemented across 17 packages under `src/pdac_circuit`, comprising roughly 130
modules, with 51 analysis entrypoints in `scripts/`. No external pretrained model supplies candidate features,
pseudo-labels, or training targets at any stage. Frozen third-party predictors are permitted in exactly one
role, as hash-locked evaluation baselines, and never as an input to a design decision.

<figure>
  <img src="{{ '/images/fig8_architecture.png' | relative_url }}" alt="Block diagram of the eight-module pipeline and its data sources">
  <figcaption><b>Figure 9.</b> Pipeline organisation. The five modules on the main path run left to right,
  each drawing directly on the real data corpora above them. Module VII supplies generated promoter libraries
  into the parts stage, and Module VIII fed the target-prioritisation stage before its predictive claim was
  withdrawn, which is why that edge is drawn dashed. Any stage lacking the evidence to proceed returns a
  certified negative rather than a default value.</figcaption>
</figure>

<div class="tablewrap">
<table>
<thead><tr><th>Module</th><th>Package</th><th>Function</th></tr></thead>
<tbody>
<tr><td>I</td><td><code>targeting/</code></td><td>Ranks PDAC driver transcription factors using TCGA-PAAD against GTEx expression, the IntOGen and NCG driver catalogues, and Moffitt subtype signatures, combined by multi-criteria decision analysis in <code>prioritize.py</code></td></tr>
<tr><td>II</td><td><code>parts/</code></td><td>Holds the trained promoter-strength and enhancer-activity models and the CRISPRi repressor selection in <code>select.py</code></td></tr>
<tr><td>III</td><td><code>circuit/</code></td><td>Builds AND and NOT logic with feedback, then analyses viability and robustness through Boolean-network and Hill-ODE treatments in <code>stability.py</code></td></tr>
<tr><td>IV</td><td><code>seqopt/</code></td><td>Optimises sequence for GC content, cryptic splice sites, restriction sites, codon adaptation by Viterbi decoding, and 5′ structure</td></tr>
<tr><td>V</td><td><code>grna/</code></td><td>Scans for PAMs, applies the trained on-target model, and scores off-targets by CFD and MIT, with the genome-wide search in <code>genome_offtarget.py</code></td></tr>
<tr><td>VI</td><td><code>scoring/</code></td><td>Resolves efficacy, specificity, robustness and safety objectives by NSGA-II Pareto optimisation</td></tr>
<tr><td>VII</td><td><code>generate/</code></td><td>Contains the promoter WGAN-GP and its evaluation in <code>evaluate.py</code></td></tr>
<tr><td>VIII</td><td><code>attractor/</code></td><td>Implements the regulatory attractor-control dynamics whose central claim is retracted, as set out on the <a href="{{ '/validation/' | relative_url }}">validation page</a></td></tr>
</tbody>
</table>
</div>

Supporting packages carry the machinery that the eight modules rely on. `harness/` holds the shared training
loop, the split logic and the fixture system. `data/` holds the loaders and interval indexing. `stats/` holds
the metrics, the permutation and conformal routines, and the certification lattice. `chromatin/` is the
largest package at 30 modules and covers the long-range chromatin model and its data handling.

## Trained models

### gRNA on-target efficiency, module V

Training data combines Doench-2016 Rule Set 2, which contributes 5,310 guides across 17 genes with a
drug-gene rank endpoint, and Kim et al. 2019, which contributes 12,832 high-throughput SpCas9 guides with a
background-subtracted indel frequency endpoint. Both use the same 4 + 20 + 3 + 3 nucleotide 30-mer context.
Assembly happens in `src/pdac_circuit/grna/datamodule.py`.

Because the two endpoints are not the same quantity, Kim's indel percentages are rank-normalised within
their own dataset before the sets are pooled. Spearman is the reporting metric, so a monotone within-dataset
transform is the correct way to combine them, and it avoids the mistake of treating a rank in one assay as
equivalent to a percentage in another.

The model itself is an ensemble. A gradient-boosted tree reads engineered Rule-Set-2 style features covering
position-wise nucleotide identity, dinucleotide counts, GC content and melting temperature, while a
convolutional network reads the 30-mer one-hot encoding directly. Evaluation uses the gene-grouped split in
`training.py`, so no gene appears on both sides and guide-neighbourhood leakage is prevented. Kim's synthetic
targets receive disjoint pseudo-groups so they can never share a group with a Doench gene. The headline
figure is measured on the held-out genes CCDC101, CD15 and CD45, which together supply 688 guides, and that
set is held identical across every comparison reported anywhere in the project.

### Promoter strength, module II

Training data is FANTOM5 CAGE, giving 209,374 peaks on standard chromosomes once lifted to hg38 windows, each
labelled by log10 mean TPM across samples. Assembly is in `src/pdac_circuit/parts/datamodule.py`.

The architecture pairs a dilated convolutional network over 1,000 bp of one-hot sequence, supplemented with
GC and CpG auxiliary scalars, against a gradient-boosted tree over 4-mer frequency features. Predictions are
mapped into the unit interval through the training CDF, which means the reported strength is a rank within
the training distribution rather than a physical unit, and that distinction is preserved wherever the value
is consumed downstream in `parts/select.py`.

Evaluation holds chr8 and chr9 for test and chr7 for validation, with everything else used for training. The
ensemble weight is chosen by grid search on validation only.

### Enhancer activity, module II

Positives are ENCODE pancreas ATAC peaks that intersect H3K27ac, labelled with the H3K27ac signal. Negatives
come in two kinds, because a classifier trained only against easy negatives learns an easy problem. Hard
negatives are accessible regions carrying no H3K27ac, and easy negatives are random genomic background drawn
from outside any ATAC peak. The model is a multitask convolutional network over 2,000 bp with a
classification head and a signal-regression head trained jointly, which lets the signal head act as an
auxiliary constraint on the representation rather than a separate model.

### Promoter generator, module VII

The generator is trained on the top-activity quartile of FANTOM5 promoters, giving 52,342 real sequences of
1,024 bp. It is a WGAN-GP with gradient penalty of 10 and five critic steps per generator step, run for 2,500
generator iterations with early stopping on the best 4-mer divergence.

Certification is deliberately two-sided and was fixed before training. Generated promoters must match real
4-mer composition better than random DNA, requiring Jensen-Shannon divergence at or below 0.05 and strictly
below the random baseline, and the library must contain a strong selectable tail, requiring a 90th-percentile
predicted strength at or above 0.7. Median strength uplift is reported but deliberately not gated, and the
reason matters. A faithful generator reproduces the full real promoter distribution, which is weak-heavy, so
its median necessarily sits near random. Gating on the median would therefore reward a generator that had
learned the wrong distribution. What the pipeline actually consumes is the strongest sequence in a generated
library, which is why the tail is the gated quantity.

## Evaluation discipline

These rules are what make numbers comparable across the project, and several of them exist because an earlier
version of the work violated them.

Baselines are always re-scored rather than quoted. When a model is retrained, the previously deployed model
is evaluated again on the identical held-out set, because a historical reported figure can differ for reasons
unrelated to the change under test.

Hyperparameters that touch the reported number are selected on validation. Ensemble weights are chosen on a
held-out validation split that never overlaps test. Where an older weight had been chosen with knowledge of
test performance, it was replaced even though doing so lowered the headline figure slightly.

New data is diagnosed before it is integrated. The Doench to Kim cross-dataset test is the worked example,
and it was run with the explicit understanding that a result near chance would have prevented the merge.

Thresholds are pre-registered. Margins live in a registry written before training, so a result either clears
a bar committed in advance or is reported as not clearing it. The gRNA Spearman margin, the promoter and
enhancer margins, the generator's realism and tail bounds, and the guide-specificity minimum are all fixed
there.

Deployed weights are provenance-locked. Each model ships a frozen set of real test rows together with their
CPU predictions, handled by `src/pdac_circuit/harness/fixtures.py`. Reloading the checkpoint must reproduce
those predictions to within 1e-4 or the model fails its gate, which catches silent weight corruption and
environment drift. All four deployed models currently reproduce exactly.

Absence of evidence is reported as absence rather than as a default value. The locus-neighbourhood off-target
search cannot establish specificity, so it never populates the specificity field at all. Only a genome-wide
scan does, and every other path fails safe to maximum risk. This is why disabling the genome-wide search
yields a certified negative instead of an apparently clean guide.

## Reproducing the analysis

The site's figures are generated from the pipeline's result files rather than transcribed, so a published
number cannot drift from the data behind it.

```
python scripts/make_figures.py
python scripts/build_pages.py
```

The first script reads the model manifests and result JSONs and writes each figure as a 300 dpi PNG together
with a vector PDF. The second converts the project's primary documents into the pages under
[full reports]({{ '/reports/' | relative_url }}) and regenerates the
[evaluation tables]({{ '/evaluation/' | relative_url }}) from the same sources.
