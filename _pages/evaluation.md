---
layout: default
title: "Evaluation"
subtitle: "Exact held-out numbers for every model, split and comparison."
description: "Complete evaluation tables read directly from the pipeline result files."
permalink: /evaluation/
---

Every figure on this page is read directly from the pipeline's result files. Where a number is a
held-out measurement, the split and its size are stated with it.

## Deployed models

<div class="tablewrap">
<table>
<thead><tr><th>Model</th><th>Metric</th><th class="num">Value</th><th>Held-out split</th><th class="num">n test</th></tr></thead>
<tbody>
<tr><td>gRNA on-target, ensemble</td><td>Spearman</td><td class="num">0.6571</td><td>gene-grouped (CCDC101, CD15, CD45)</td><td class="num">688</td></tr>
<tr><td>gRNA on-target, CNN</td><td>Spearman</td><td class="num">0.6167</td><td>same</td><td class="num">688</td></tr>
<tr><td>gRNA on-target, GBM</td><td>Spearman</td><td class="num">0.6504</td><td>same</td><td class="num">688</td></tr>
<tr><td>Promoter, ensemble</td><td>Spearman</td><td class="num">0.5275</td><td>chromosome-held-out (chr8, chr9)</td><td class="num">16,940</td></tr>
<tr><td>Promoter, CNN</td><td>Spearman</td><td class="num">0.5247</td><td>same</td><td class="num">16,940</td></tr>
<tr><td>Promoter, tree model</td><td>Spearman</td><td class="num">0.5081</td><td>same</td><td class="num">16,940</td></tr>
<tr><td>Enhancer, classification</td><td>AUROC</td><td class="num">0.8375</td><td>chromosome-held-out (chr8, chr9)</td><td class="num">13,425</td></tr>
<tr><td>Enhancer, signal head</td><td>Spearman</td><td class="num">0.5162</td><td>active rows of the same test set</td><td class="num">—</td></tr>
<tr><td>Generator, realism</td><td>4-mer JS</td><td class="num">0.0123</td><td>fixed 2,000-promoter real reference</td><td class="num">1,500</td></tr>
<tr><td>Generator, selectable tail</td><td>p90 strength</td><td class="num">0.9918</td><td>same</td><td class="num">1,500</td></tr>
</tbody>
</table>
</div>

Ensemble weights, chosen on validation and never on test: gRNA 0.40 CNN /
0.60 GBM; promoter 0.86 CNN /
0.14 tree. Permutation p for the gRNA and promoter models is
0.000999, the floor of a 1,000-permutation test.

## Before and after removing the data caps

<div class="tablewrap">
<table>
<thead><tr><th>Model</th><th class="num">Capped</th><th class="num">Full</th><th class="num">Δ</th><th class="num">n train</th><th>Deployed</th></tr></thead>
<tbody>
<tr><td>gRNA on-target</td><td class="num">0.4938</td><td class="num up">0.6571</td><td class="num up">+0.1633</td><td class="num">15,497</td><td>yes</td></tr>
<tr><td>Promoter</td><td class="num">0.5199</td><td class="num up">0.5275</td><td class="num up">+0.0075</td><td class="num">181,428</td><td>yes</td></tr>
<tr><td>Enhancer</td><td class="num">0.8087</td><td class="num up">0.8147</td><td class="num up">+0.0060</td><td class="num">135,402</td><td>yes</td></tr>
<tr><td>Generator, 4-mer JS</td><td class="num">0.0088</td><td class="num dn">0.0123</td><td class="num dn">+0.0034</td><td class="num">52,342</td><td rowspan="2">yes, on tail</td></tr>
<tr><td>Generator, p90 tail</td><td class="num">0.9369</td><td class="num up">0.9918</td><td class="num up">+0.0548</td><td class="num">52,342</td></tr>
</tbody>
</table>
</div>

The generator is the one case where the two axes disagree: divergence rose slightly while the selectable
tail improved substantially. It is deployed on the tail, which is the axis the pipeline consumes, and both
versions clear the pre-registered certification.

## Cross-dataset and cross-domain generalisation

<div class="tablewrap">
<table>
<thead><tr><th>Test</th><th class="num">Score</th><th>Interpretation</th></tr></thead>
<tbody>
<tr><td>Train Doench (17 genes) → test Kim (12,832 guides)</td><td class="num">0.5918</td><td>Higher than the within-Doench held-out; justified pooling</td></tr>
<tr><td>Kim within-dataset ceiling</td><td class="num">0.7217</td><td>The larger, cleaner library supports a higher ceiling</td></tr>
<tr><td>Enhancer, pancreas → pancreas</td><td class="num">0.8150</td><td>Within-domain reference</td></tr>
<tr><td>Enhancer, pancreas → PANC-1</td><td class="num">0.8349</td><td>Forward transfer, above its own domain</td></tr>
<tr><td>Enhancer, PANC-1 → pancreas</td><td class="num">0.7899</td><td>Reverse transfer; asymmetry favours the multi-donor source</td></tr>
<tr><td>Enhancer, merged training → pancreas</td><td class="num dn">0.8096</td><td>-0.0054 against pancreas-only; not deployed</td></tr>
</tbody>
</table>
</div>

## Scaling curves, every point

Independent trainings at each size, all scored on the same fixed held-out test. Run-to-run variation is
roughly 0.005, so the trend is the result rather than any single point.

<div class="tablewrap">
<table>
<thead><tr><th class="num">Promoter n train</th><th class="num">CNN</th><th class="num">Tree</th><th class="num">Ensemble</th><th class="num">weight CNN</th></tr></thead>
<tbody>
<tr><td class='num'>10,000</td><td class='num'>0.4892</td><td class='num'>0.4691</td><td class='num'>0.4982</td><td class='num'>0.49</td></tr>
<tr><td class='num'>20,000</td><td class='num'>0.4772</td><td class='num'>0.4946</td><td class='num'>0.5013</td><td class='num'>0.25</td></tr>
<tr><td class='num'>40,000</td><td class='num'>0.5028</td><td class='num'>0.4930</td><td class='num'>0.5125</td><td class='num'>0.65</td></tr>
<tr><td class='num'>80,000</td><td class='num'>0.5142</td><td class='num'>0.5028</td><td class='num'>0.5193</td><td class='num'>0.73</td></tr>
<tr><td class='num'>120,000</td><td class='num'>0.5296</td><td class='num'>0.5050</td><td class='num'>0.5315</td><td class='num'>0.88</td></tr>
<tr><td class='num'>181,428</td><td class='num'>0.5279</td><td class='num'>0.5081</td><td class='num'>0.5325</td><td class='num'>0.77</td></tr>
</tbody>
</table>
</div>

<div class="tablewrap">
<table>
<thead><tr><th class="num">Enhancer n train</th><th class="num">AUROC</th></tr></thead>
<tbody>
<tr><td class='num'>20,000</td><td class='num'>0.8049</td></tr>
<tr><td class='num'>40,000</td><td class='num'>0.8037</td></tr>
<tr><td class='num'>80,000</td><td class='num'>0.8059</td></tr>
<tr><td class='num'>135,402</td><td class='num'>0.8121</td></tr>
</tbody>
</table>
</div>

## Promoter H3K27ac, per target

PDAC (PANC-1) against healthy pancreas on ENCODE fold-change-over-control tracks, TSS ±2000 bp, GRCh38.
Target mean +0.9193 log2 against background -0.0914 across
1,655 loci; 70% of targets gain signal against
45.6% of background; Mann-Whitney one-sided p = 0.002239.

<div class="tablewrap">
<table>
<thead><tr><th>Target</th><th class="num">PDAC fold-change</th><th class="num">Healthy fold-change</th><th class="num">log2 residual</th></tr></thead>
<tbody>
<tr><td>HOXA3</td><td class='num'>5.772</td><td class='num'>0.000</td><td class='num up'>+5.8757</td></tr>
<tr><td>FOSL1</td><td class='num'>20.044</td><td class='num'>2.849</td><td class='num up'>+2.7719</td></tr>
<tr><td>MYBL2</td><td class='num'>5.550</td><td class='num'>1.137</td><td class='num up'>+2.1913</td></tr>
<tr><td>SMAD3</td><td class='num'>17.501</td><td class='num'>3.948</td><td class='num up'>+2.1205</td></tr>
<tr><td>ZNF528</td><td class='num'>6.557</td><td class='num'>1.583</td><td class='num up'>+1.9835</td></tr>
<tr><td>E2F1</td><td class='num'>5.958</td><td class='num'>1.651</td><td class='num up'>+1.7907</td></tr>
<tr><td>AHR</td><td class='num'>2.277</td><td class='num'>0.898</td><td class='num up'>+1.2519</td></tr>
<tr><td>ZNF331</td><td class='num'>10.033</td><td class='num'>4.295</td><td class='num up'>+1.2051</td></tr>
<tr><td>SETDB1</td><td class='num'>13.997</td><td class='num'>7.042</td><td class='num up'>+0.9811</td></tr>
<tr><td>AGR2</td><td class='num'>1.528</td><td class='num'>0.757</td><td class='num up'>+0.9259</td></tr>
<tr><td>ZNF790</td><td class='num'>6.938</td><td class='num'>4.003</td><td class='num up'>+0.7783</td></tr>
<tr><td>ATM</td><td class='num'>16.865</td><td class='num'>10.737</td><td class='num up'>+0.6466</td></tr>
<tr><td>SF3B1</td><td class='num'>19.602</td><td class='num'>16.824</td><td class='num up'>+0.2193</td></tr>
<tr><td>ZNF93</td><td class='num'>7.587</td><td class='num'>7.216</td><td class='num up'>+0.0714</td></tr>
<tr><td>BRCA2</td><td class='num'>11.673</td><td class='num'>13.071</td><td class='num dn'>-0.1618</td></tr>
<tr><td>SOX13</td><td class='num'>7.470</td><td class='num'>8.812</td><td class='num dn'>-0.2355</td></tr>
<tr><td>KMT2C</td><td class='num'>6.603</td><td class='num'>9.532</td><td class='num dn'>-0.5230</td></tr>
<tr><td>FAM83A</td><td class='num'>1.165</td><td class='num'>1.763</td><td class='num dn'>-0.5586</td></tr>
<tr><td>ZNF85</td><td class='num'>4.484</td><td class='num'>7.408</td><td class='num dn'>-0.7118</td></tr>
<tr><td>GATA6</td><td class='num'>1.408</td><td class='num'>7.002</td><td class='num dn'>-2.2356</td></tr>
</tbody>
</table>
</div>

## Adversarial validation of the attractor claim

Across 419 genes with 31 essential positives, a positive rate of
0.074.

<div class="tablewrap">
<table>
<thead><tr><th>Statistic</th><th class="num">Value</th></tr></thead>
<tbody>
<tr><td>AUC, attractor collapse</td><td class="num dn">0.5471</td></tr>
<tr><td>AUC, network degree</td><td class="num up">0.6290</td></tr>
<tr><td>AUC, eigenvector centrality</td><td class="num">0.5843</td></tr>
<tr><td>Δ AUC, collapse minus degree</td><td class="num dn">-0.0819</td></tr>
<tr><td>95% CI, paired bootstrap</td><td class="num">[-0.199, 0.029]</td></tr>
<tr><td>p, two-sided</td><td class="num">0.1468</td></tr>
<tr><td>PR-AUC, collapse</td><td class="num">0.0979</td></tr>
<tr><td>PR-AUC, degree</td><td class="num">0.1311</td></tr>
<tr><td>PR-AUC baseline, positive rate</td><td class="num">0.0740</td></tr>
<tr><td>Precision at 10 / 20 / 50</td><td class="num">0.10 / 0.15 / 0.12</td></tr>
<tr><td>Partial Spearman given degree, expression, variance</td><td class="num">0.0284 (p = 0.563)</td></tr>
<tr><td>Cross-validated AUC, covariates only</td><td class="num">0.6534</td></tr>
<tr><td>Cross-validated AUC, plus collapse</td><td class="num">0.6524</td></tr>
</tbody>
</table>
</div>

Adding the collapse score to a model already containing degree, expression and variance changes
cross-validated AUC by -0.0010.
