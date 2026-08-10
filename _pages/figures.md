---
layout: default
title: "Figures"
subtitle: "Every figure on the site, with its caption and a vector copy."
description: "Complete figure list for the PDAC chromatin-circuit project."
permalink: /figures/
---

Each figure is written as a 300 dpi raster together with a vector copy suitable for print. The captions below are the captions carried on the pages where the figures appear, extracted at build time rather than rewritten here.

<ul class="doclist figindex">
<li><a href="#fig1">Figure 1</a> <span class="l">Held-out performance before and after removing training-data caps, for four models</span></li>
<li><a href="#fig2">Figure 2</a> <span class="l">Per-target log2 H3K27ac fold-change residual between PDAC and healthy pancreas</span></li>
<li><a href="#fig3">Figure 3</a> <span class="l">Attractor collapse versus network degree as predictors of essentiality, with confidence interval on the difference</span></li>
<li><a href="#fig4">Figure 4</a> <span class="l">CNN, GBM and ensemble Spearman before and after adding the Kim-2019 dataset</span></li>
<li><a href="#fig5">Figure 5</a> <span class="l">Held-out performance versus training-set size for the promoter and enhancer models</span></li>
<li><a href="#fig6">Figure 6</a> <span class="l">Enhancer AUROC within domain and across the healthy-pancreas to PDAC boundary in both directions</span></li>
<li><a href="#fig7">Figure 7</a> <span class="l">Heatmap of multi-omic evidence layers across the prioritised target genes</span></li>
<li><a href="#fig8">Figure 8</a> <span class="l">Schematic of a designed CRISPRi circuit targeting GATA6 with negative feedback</span></li>
<li><a href="#fig9">Figure 9</a> <span class="l">Block diagram of the eight-module pipeline and its data sources</span></li>
<li><a href="#fig10">Figure 10</a> <span class="l">Attractor collapse versus degree and eigenvector centrality, with a confidence interval on the difference</span></li>
<li><a href="#fig11">Figure 11</a> <span class="l">Four-panel summary of the circuit design campaign</span></li>
</ul>

<figure id="fig1">
  <img src="{{ '/images/fig1_scaleup.png' | relative_url }}" alt="Held-out performance before and after removing training-data caps, for four models">
  <figcaption><b>Figure 1.</b> Held-out performance before and after the caps were removed. Each comparison is like for like, because the previously deployed model is re-scored on the identical held-out set rather than compared against its historical reported figure. Grey hatched bars are the capped models and blue bars are the full-data models. The shaded band is the run-to-run variation of an independent retraining, so a bar ending inside it has not been shown to differ from its predecessor. The y-scales are independent. Produced by <code>scripts/promoter_scaleup.py</code>, <code>scripts/enhancer_scaleup.py</code>, <code>scripts/grna_cnn_kim_retrain.py</code> and <code>scripts/promoter_gan_scaleup.py</code>.
  <span class="figmeta">Appears in <a href="{{ '/' | relative_url }}">Overview</a>. Vector copy <a href="{{ '/images/fig1_scaleup.pdf' | relative_url }}">fig1_scaleup.pdf</a>.</span></figcaption>
</figure>

<figure id="fig2">
  <img src="{{ '/images/fig4_h3k27ac.png' | relative_url }}" alt="Per-target log2 H3K27ac fold-change residual between PDAC and healthy pancreas">
  <figcaption><b>Figure 2.</b> Promoter H3K27ac in PDAC against healthy pancreas for the 20 prioritised targets. The target mean sits at +0.919 log2 while the background mean sits at −0.091, and 70 percent of targets gain signal against 46 percent of background. Measured on ENCODE fold-change-over-control tracks by <code>scripts/pdac_residual_foldchange.py</code>, with the parameter sweeps in <code>scripts/h3k27ac_fragility.py</code>, <code>scripts/h3k27ac_pseudocount.py</code> and <code>scripts/h3k27ac_window_and_loci.py</code>.
  <span class="figmeta">Appears in <a href="{{ '/' | relative_url }}">Overview</a>. Vector copy <a href="{{ '/images/fig4_h3k27ac.pdf' | relative_url }}">fig4_h3k27ac.pdf</a>.</span></figcaption>
</figure>

<figure id="fig3">
  <img src="{{ '/images/fig5_rac_validation.png' | relative_url }}" alt="Attractor collapse versus network degree as predictors of essentiality, with confidence interval on the difference">
  <figcaption><b>Figure 3.</b> Attractor collapse reaches AUC 0.547 against plain network degree at 0.629, measured across 419 genes carrying 31 essential positives. The paired-bootstrap confidence interval on the difference spans zero, and once degree is controlled for, the score contributes nothing further. The claim is retracted.
  <span class="figmeta">Appears in <a href="{{ '/' | relative_url }}">Overview</a>. Vector copy <a href="{{ '/images/fig5_rac_validation.pdf' | relative_url }}">fig5_rac_validation.pdf</a>.</span></figcaption>
</figure>

<figure id="fig4">
  <img src="{{ '/images/fig3_grna_components.png' | relative_url }}" alt="CNN, GBM and ensemble Spearman before and after adding the Kim-2019 dataset">
  <figcaption><b>Figure 4.</b> Component breakdown on the identical 688 held-out-gene guides. The CNN was the binding constraint. At 0.392 on 17 genes it was close to useless, and it had been down-weighted to 0.20 in the ensemble largely to stop it doing harm. On 18,142 guides it reaches 0.617, and the deployed ensemble is a balanced 0.40 CNN against 0.60 GBM. Generated by <code>scripts/grna_cnn_kim_retrain.py</code>.
  <span class="figmeta">Appears in <a href="{{ '/results/' | relative_url }}">Results</a>. Vector copy <a href="{{ '/images/fig3_grna_components.pdf' | relative_url }}">fig3_grna_components.pdf</a>.</span></figcaption>
</figure>

<figure id="fig5">
  <img src="{{ '/images/fig2_scaling_curves.png' | relative_url }}" alt="Held-out performance versus training-set size for the promoter and enhancer models">
  <figcaption><b>Figure 5.</b> Performance against training-set size on the fixed chr8 and chr9 test. The promoter ensemble climbs at every one of six sizes without a reversal, by 0.035 in total across an eighteen-fold increase in data, then flattens over the final step. A strictly increasing run of six points has probability 1/6! = 0.0014 under a null of no trend, which is what makes the promoter result credible where its single delta does not. The enhancer curve spans only 0.007 in total and is not monotone. These are independent trainings, so run-to-run variation of roughly 0.005 is expected and the trend rather than any single point is the result.
  <span class="figmeta">Appears in <a href="{{ '/results/' | relative_url }}">Results</a>. Vector copy <a href="{{ '/images/fig2_scaling_curves.pdf' | relative_url }}">fig2_scaling_curves.pdf</a>.</span></figcaption>
</figure>

<figure id="fig6">
  <img src="{{ '/images/fig6_cross_domain.png' | relative_url }}" alt="Enhancer AUROC within domain and across the healthy-pancreas to PDAC boundary in both directions">
  <figcaption><b>Figure 6.</b> A healthy-pancreas model predicts PANC-1 PDAC enhancers at AUROC 0.835, which is higher than its score on its own domain, and the reverse direction holds at 0.790. The asymmetry follows from the composition of the two training sets, since the multi-donor pancreas panel is the more diverse source and generalises to a single cell line more readily than one cell line generalises to a panel. Computed by <code>scripts/enhancer_panc1_augment.py</code> and <code>scripts/enhancer_scaling_curve.py</code>.
  <span class="figmeta">Appears in <a href="{{ '/results/' | relative_url }}">Results</a>. Vector copy <a href="{{ '/images/fig6_cross_domain.pdf' | relative_url }}">fig6_cross_domain.pdf</a>.</span></figcaption>
</figure>

<figure id="fig7">
  <img src="{{ '/images/fig7_evidence_heatmap.png' | relative_url }}" alt="Heatmap of multi-omic evidence layers across the prioritised target genes">
  <figcaption><b>Figure 7.</b> Multi-omic evidence for each prioritised target, with every layer standardised within its own column so that quantities on different scales can be read side by side. Red indicates a higher value within a layer and blue a lower one. Hatched cells are layers in which that gene was not measured, and two columns are hatched throughout, which records that promoter methylation and the ATAC residual are unavailable for this target set rather than that they were zero. The pattern is deliberately uneven, since no single layer nominates a target and the prioritisation depends on agreement across several. Assembled by <code>scripts/rac_target_dossiers.py</code>.
  <span class="figmeta">Appears in <a href="{{ '/results/' | relative_url }}">Results</a>. Vector copy <a href="{{ '/images/fig7_evidence_heatmap.pdf' | relative_url }}">fig7_evidence_heatmap.pdf</a>.</span></figcaption>
</figure>

<figure id="fig8">
  <img src="{{ '/images/fig9_circuit.png' | relative_url }}" alt="Schematic of a designed CRISPRi circuit targeting GATA6 with negative feedback">
  <figcaption><b>Figure 8.</b> One designed circuit from the classical-subtype run. Panel a is the delivered construct and its action at the GATA6 locus on chr18, with both cassettes drawn in SBOL genetic notation, the dCas9-KRAB effector loaded on its guide RNA at the target, a blunt head marking transcriptional repression, and the dashed loop marking negative feedback. The part values beneath it are the measured ones. Panel b is not a sketch. It is the ODE in <code>pdac_circuit.circuit.ode</code> integrated on this circuit's own rate constants, which reproduce the published knockdown of 0.647 exactly, showing the overshoot and settling the feedback produces. Diagram created with BioRender; assembled by <code>scripts/make_circuit_figure.py</code>.
  <span class="figmeta">Appears in <a href="{{ '/results/' | relative_url }}">Results</a>. Vector copy <a href="{{ '/images/fig9_circuit.pdf' | relative_url }}">fig9_circuit.pdf</a>.</span></figcaption>
</figure>

<figure id="fig9">
  <img src="{{ '/images/fig8_architecture.png' | relative_url }}" alt="Block diagram of the eight-module pipeline and its data sources">
  <figcaption><b>Figure 9.</b> Pipeline organisation. The five modules on the main path run left to right, each drawing directly on the real data corpora above them. Module VII supplies generated promoter libraries into the parts stage, and Module VIII fed the target-prioritisation stage before its predictive claim was withdrawn, which is why that edge is drawn dashed. Any stage lacking the evidence to proceed returns a certified negative rather than a default value.
  <span class="figmeta">Appears in <a href="{{ '/methods/' | relative_url }}">Methods</a>. Vector copy <a href="{{ '/images/fig8_architecture.pdf' | relative_url }}">fig8_architecture.pdf</a>.</span></figcaption>
</figure>

<figure id="fig10">
  <img src="{{ '/images/fig5_rac_validation.png' | relative_url }}" alt="Attractor collapse versus degree and eigenvector centrality, with a confidence interval on the difference">
  <figcaption><b>Figure 10.</b> Across 419 genes carrying 31 essential positives, attractor collapse reaches AUC 0.547 while plain network degree reaches 0.629. The paired-bootstrap 95 percent confidence interval on the difference spans zero.
  <span class="figmeta">Appears in <a href="{{ '/validation/' | relative_url }}">Validation</a>. Vector copy <a href="{{ '/images/fig5_rac_validation.pdf' | relative_url }}">fig5_rac_validation.pdf</a>.</span></figcaption>
</figure>

<figure id="fig11">
  <img src="{{ '/images/fig10_design_campaign.png' | relative_url }}" alt="Four-panel summary of the circuit design campaign">
  <figcaption><b>Figure 11.</b> The design campaign in full. Composite score across the whole enumeration separated by single and paired targets, the efficacy against safety trade-off with the non-dominated set marked, simulated knockdown against swept promoter strength shown as median and interquartile range, and the best circuit found for each target. Generated by <code>scripts/make_campaign_figure.py</code> from <code>results/circuit_design_campaign_all.jsonl.gz</code>.
  <span class="figmeta">Appears in <a href="{{ '/circuits/' | relative_url }}">Circuits</a>. Vector copy <a href="{{ '/images/fig10_design_campaign.pdf' | relative_url }}">fig10_design_campaign.pdf</a>.</span></figcaption>
</figure>
