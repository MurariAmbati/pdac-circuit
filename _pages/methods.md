---
layout: default
title: Methods
subtitle: The eight modules, how each model is trained and evaluated, and the discipline applied to every number.
permalink: /methods/
---

## Pipeline architecture

Eight modules, built from scratch in Python. No external pretrained model supplies candidate features,
pseudo-labels, or training targets at any point; frozen third-party predictors are permitted only as
hash-locked evaluation baselines, never as inputs.

<div class="tablewrap">
<table>
<thead><tr><th>Module</th><th>Package</th><th>Function</th></tr></thead>
<tbody>
<tr><td>I</td><td><code>targeting/</code></td><td>Rank PDAC-driver transcription factors from TCGA-PAAD versus GTEx expression, IntOGen and NCG driver catalogues, and Moffitt subtype signatures, combined by multi-criteria decision analysis</td></tr>
<tr><td>II</td><td><code>parts/</code></td><td>Trained promoter-strength and enhancer-activity models; CRISPRi repressor selection</td></tr>
<tr><td>III</td><td><code>circuit/</code></td><td>AND / NOT logic with feedback; Boolean-network and Hill-ODE viability and robustness analysis</td></tr>
<tr><td>IV</td><td><code>seqopt/</code></td><td>Sequence optimisation: GC content, cryptic splice sites, restriction sites, codon adaptation by Viterbi, 5′ structure</td></tr>
<tr><td>V</td><td><code>grna/</code></td><td>PAM scanning, trained on-target efficiency model, CFD and MIT off-target scoring</td></tr>
<tr><td>VI</td><td><code>scoring/</code></td><td>Efficacy, specificity, robustness and safety objectives resolved by NSGA-II Pareto optimisation</td></tr>
<tr><td>VII</td><td><code>generate/</code></td><td>Trained promoter WGAN-GP generating novel synthetic promoters</td></tr>
<tr><td>VIII</td><td><code>attractor/</code></td><td>Regulatory attractor-control dynamics — see <a href="{{ '/validation/' | relative_url }}">Validation</a> for its retraction</td></tr>
</tbody>
</table>
</div>

## Trained models

### gRNA on-target efficiency (Module V)

**Data.** Doench-2016 Rule Set 2 (5,310 guides, 17 genes, drug-gene rank endpoint) merged with Kim et al.
2019 (12,832 high-throughput SpCas9 guides, background-subtracted indel frequency). Both use the identical
(4 + 20 + 3 + 3) 30-mer context format.

**Pooling two endpoints.** The two datasets measure different quantities, so Kim's indel percentages are
rank-normalised *within dataset* before pooling. Spearman is the reporting metric, so a monotone
within-dataset transform is the appropriate way to combine them; the cross-dataset generalisation test
described in Results is what justified pooling at all.

**Architecture.** A gradient-boosted tree over engineered Rule-Set-2 style features — position-wise
nucleotide one-hot, dinucleotide counts, GC content, melting temperature — ensembled with a sequence CNN
reading the 30-mer one-hot directly.

**Evaluation.** Gene-grouped split so no gene appears in both train and test, which prevents
guide-neighbourhood leakage. Kim's synthetic targets receive disjoint pseudo-groups so they can never share
a group with a Doench gene. The reported figure is on the held-out Doench genes CCDC101, CD15 and CD45
(688 guides), held identical across every comparison.

### Promoter strength (Module II)

**Data.** FANTOM5 CAGE peaks lifted to hg38 windows, 209,374 on standard chromosomes, labelled by
log10 mean TPM across samples.

**Architecture.** A dilated sequence CNN over 1,000 bp one-hot with GC and CpG auxiliary scalars, ensembled
with a gradient-boosted tree over 4-mer frequency features. Output is mapped to [0, 1] through the training
CDF, so the reported "strength" is a rank within the training distribution rather than a physical unit.

**Evaluation.** Chromosome-held-out: chr8 and chr9 are test, chr7 validation, everything else train. The
ensemble weight is chosen by grid search on validation.

### Enhancer activity (Module II)

**Data.** ENCODE pancreas ATAC-seq peaks. Active examples are ATAC peaks intersecting H3K27ac, labelled with
H3K27ac signal; negatives are hard negatives (accessible but unmarked) plus random genomic background
excluded from any ATAC peak.

**Architecture.** A multitask CNN over 2,000 bp with two heads — active/inactive classification and
H3K27ac signal regression — trained jointly.

**Evaluation.** Same chromosome-held-out protocol. AUROC on the classification head is the headline; signal
Spearman on active regions is reported alongside.

### Promoter generator (Module VII)

**Data.** The top-activity quartile of FANTOM5 promoters, 52,342 real 1,024 bp sequences.

**Architecture.** WGAN-GP with gradient penalty λ = 10 and five critic steps per generator step, trained for
2,500 generator iterations with best-4-mer-divergence early stopping.

**Certification.** Pre-registered and two-sided: generated promoters must match real 4-mer composition better
than random DNA (Jensen–Shannon ≤ 0.05 *and* below random), and the library must contain a strong selectable
tail (90th-percentile predicted strength ≥ 0.7). Median strength uplift is reported but deliberately not
gated — a faithful generator reproduces the full, weak-heavy real promoter distribution, so its median sits
near random by construction. The value is realistic composition plus a strong tail, because the pipeline
selects the strongest generated promoter rather than a random one.

## Evaluation discipline

These rules are what make the numbers comparable across the project.

**Apples-to-apples baselines.** When a model is retrained, the previously deployed model is re-scored on the
identical held-out set rather than compared against its historical reported figure. Historical numbers can
differ for reasons unrelated to the change being tested.

**Validation-selected hyperparameters.** Ensemble weights are chosen on a held-out validation split that never
overlaps test. Where an older weight had been chosen with knowledge of test performance, it was replaced.

**Diagnose before integrating.** New data is tested for transferability before being merged. The
Doench→Kim cross-dataset score is the example: had it come back near chance, the datasets would not have been
pooled.

**Pre-registration.** Thresholds and margins live in a registry written before training, so a result either
clears a pre-committed bar or is reported as not clearing it. The gRNA Spearman margin, the promoter and
enhancer margins, the generator's realism and tail bounds, and the guide-specificity minimum are all fixed in
advance.

**Frozen predeploy fixtures.** Each deployed model ships a frozen set of real test rows together with their
CPU predictions. Reloading the checkpoint must reproduce those predictions to 1 × 10⁻⁴ or the model fails its
gate. This catches silent weight corruption and environment drift, and it is verified for all four deployed
models.

**Provenance by hash.** Every corpus is recorded with source URL, byte count and SHA-256. Raw data bytes stay
out of version control; the hashes are the provenance. Model weights are likewise excluded, with the
manifest's `weight_sha256` binding a reported metric to a specific checkpoint.

**Honest abstention.** Where evidence is absent, the pipeline returns a certified-negative rather than a
default value. The locus-neighbourhood off-target search cannot establish specificity, so it never populates
the specificity field; only a genome-wide scan does, and every other path fails safe to maximum risk.

## Reproducing the figures

Every figure on this site is generated from the pipeline's result files rather than transcribed, so the
published numbers cannot drift from the data:

```
python scripts/make_figures.py
```

The script reads the model manifests and result JSONs directly and writes both PNG and vector PDF at 300 dpi.
