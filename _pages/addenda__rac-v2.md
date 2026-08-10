---
layout: default
title: "Addendum: attractor model v2"
subtitle: "The rebuilt directed-motif substrate, its gate, and the supervised ceiling."
description: "The rebuilt directed-motif substrate, its gate, and the supervised ceiling."
permalink: /addenda/rac-v2/
group: addenda
order: 4
---

This addendum documents an attempt to *rebuild* the Regulatory Attractor
Control method after [ADDENDUM_DYNAMICS.md]({{ '/addenda/dynamics/' | relative_url }}) §7 concluded that its failure is
**structural in the substrate** — the co-expression graph does not encode essentiality beyond node
degree, so no dynamics over it can. The rebuild was executed in the honest order that diagnosis
dictates: fix the substrate first, prove it carries signal, and only then build dynamics on it. It
stopped at the substrate gate, and that outcome is the finding.

Scripts: `scripts/build_directed_grn.py`, `scripts/substrate_signal_test.py`.
Results: `results/directed_grn.npz`, `results/directed_grn.meta.json`,
`results/substrate_signal_test.json`.

---

## 1. The rebuild principle — substrate before dynamics

The retraction arc established, in order:

- §15 — collapse does not predict essentiality beyond degree (degree-matched AUC ≤ 0.49);
- §15b — the one flattering exception (a PDAC-selective hint) was a threshold/outlier artifact;
- §17 — the system is not even bistable: equilibria are unstable, 0/84 inits converge, ρ ≈ 1.02;
- §18 — raising the gain does not rescue it (0/54 convergence at every gain 4–8), so the failure is
  **structural**: *dynamics propagate only what the graph encodes*, and essentiality is not in the
  co-expression graph beyond degree.

The single most important consequence: **do not build dynamics on a substrate that has not been
shown to encode the target.** Every failure this project catalogues is a check that reported success
while structurally unable to detect the failure it was meant to catch; building an elaborate
bistable dynamical system on a graph that lacks the signal would be the same error one level up. So
the rebuild is gated: construct a better substrate, *prove* it carries essentiality signal beyond
degree, and only then build dynamics.

---

## 2. Phase 1 — a directed regulatory substrate (motif-GRN)

The current substrate is a **symmetric co-expression** adjacency (`|corr| > τ`), undirected, in
which node degree is hub-ness is essentiality almost tautologically, and motifs were annotation-only
(`motif_edges=False` in every §15/§17 fit). The natural upgrade is a **directed** topology that
encodes *who regulates whom*.

`build_directed_grn.py` constructs it: edge *i → j* carries weight `max_pwm_score(PWM_i,
promoter_j)`, the best normalised hit of transcription factor *i*'s JASPAR motif in gene *j*'s hg38
promoter (both strands, 2 kb up / 0.5 kb down). Only the 205 nodes with a JASPAR PWM have outgoing
edges; all 422 nodes have promoters. This is a genuine TF→target regulatory topology and it is
**independent of DepMap CRISPR**, so testing essentiality against it is leakage-free.

| property | value |
|---|---|
| nodes | 422 (same set as §15/§17) |
| regulators with a PWM | 205 |
| nodes with an hg38 promoter | 422 |
| nonzero motif scores | 80,292 |
| edges at score ≥ 0.90 / 0.95 | 11,331 / 9,451 |

A caveat noted honestly: short motifs saturate (90th percentile of nonzero scores = 1.0), so the
top scores are common perfect hits. Phase 2 therefore tests both threshold-free weighted strengths
and thresholded degrees, so no conclusion rests on the cutoff.

---

## 3. Phase 2 — the gate: does the directed topology beat degree?

`substrate_signal_test.py` tests every directed node-property against DepMap absolute essentiality
(Chronos > 0.4, the §15 panel: n = 419, 31 positive), with undirected co-expression degree
(AUC 0.629) as the baseline to beat. For each property: raw AUC; degree-matched concordance
controlling for co-expression degree (the §15 protocol — among genes of equal co-expression degree,
does the property still rank essential genes higher?); and partial Spearman given co-expression
degree. **Predeclared gate:** a property passes iff matched AUC > 0.55 **and** perm p < 0.05
**and** partial p < 0.05.

| property | raw AUC | degree-matched AUC | perm p | partial ρ (given degree) | p | passes? |
|---|---|---|---|---|---|---|
| **co-expression degree (baseline)** | **0.629** | — | — | — | — | — |
| out-strength (regulatory output) | 0.393 | 0.387 | 0.979 | +0.046 | 0.346 | no |
| out-degree @0.9 | 0.430 | 0.432 | 0.897 | +0.052 | 0.285 | no |
| in-strength (regulatory load) | 0.470 | 0.480 | 0.636 | −0.031 | 0.524 | no |
| in-degree @0.9 | 0.506 | 0.504 | 0.480 | −0.031 | 0.526 | no |
| directed PageRank | 0.480 | 0.444 | 0.849 | +0.009 | 0.863 | no |
| HITS hub | 0.429 | 0.431 | 0.893 | +0.053 | 0.282 | no |
| HITS authority | 0.541 | 0.535 | 0.277 | −0.016 | 0.743 | no |

**Not one directed property beats degree.** Every raw AUC is below the 0.629 baseline; no
degree-matched AUC reaches 0.55; every partial correlation given degree is non-significant
(p ≥ 0.28). The result is robust to the edge threshold because the threshold-free weighted strengths
(out-strength 0.39, in-strength 0.47) fail as badly as the thresholded degrees.

**The "master regulator" hypothesis is refuted, with the sign.** Out-strength and out-degree —
regulatory breadth, the intuition that an essential PDAC TF drives many genes — are *below* 0.5
(0.39, 0.43): if anything, regulating more targets is mildly *anti*-associated with being essential.
The only property above 0.5 is HITS authority (0.541, being bound by many regulators), and it is far
below degree and not significant after matching.

---

## 4. Decision and interpretation

**STOP.** Per the predeclared gate, the directed substrate does not encode essentiality beyond
degree, so the dynamics rebuild (Phase 3) is not warranted. Building a stability-constrained bistable
system on this topology would dress a substrate proven not to carry the signal in elaborate
machinery — the exact failure mode the rebuild was designed to avoid.

This **strengthens the §18 structural conclusion** from "co-expression degree is the ceiling" to a
sharper statement: **two independent graph topologies over these genes — undirected co-expression
and directed motif-regulatory — both fail to encode PDAC-TF essentiality beyond node degree.** The
signal is not hiding in the edge directions or in a smarter centrality; it is not in the graph
topology at all. This is consistent with the whole arc: DepMap essentiality among these TFs is
dominated by core-essential connectivity (which any graph's degree captures), and the
PDAC-*selective* vulnerability that would distinguish a real target is too sparse (§15b: ~7 selective
TFs in the panel) for topology to recover.

---

## 5. What would actually be required (and why it is not a graph tweak)

The rebuild attempt localises what genuine progress needs — and it is *information*, not
architecture:

1. **Direct perturbation data at scale.** Essentiality is what CRISPR *measures*; predicting it from
   network topology is trying to recover, from structure, a quantity that only perturbation reveals.
   A model that beats degree would need perturbation readouts (Perturb-seq, combinatorial CRISPR)
   as *input*, not as the held-out label.
2. **A causal, signed, dynamical network learned from interventions** — not motif-binding potential
   (which is promiscuous: 80k nonzero scores, saturating) and not co-expression (which is
   confounded by shared programmes). Directed edges must carry *effect*, not just *binding*.
3. **A denser, larger node set.** 419 TFs with ~30 essential positives is underpowered for any
   topology-based discrimination; the selective signal that matters lives in a handful of genes.

None of these is a reparameterisation of RAC. The honest conclusion of the rebuild is that RAC is
not one better-graph or one better-gain away from working: the class of methods — *predict
essentiality from a network over expression/motif data* — is at its ceiling here, and that ceiling
is network degree.

---

## 6. What stands, unchanged

- The **intervention gate** (REVIEW_RESPONSE.md §16) is independent of substrate and dynamics — a
  role × direction gate calibrated against TCGA copy number (11/12 roles corroborated) — and stands.
- The **directed motif-GRN itself** (`results/directed_grn.npz`) is a real, reusable artifact: a
  leakage-free TF→target topology over the PDAC TF panel. It simply does not predict essentiality —
  which is a statement about essentiality's relationship to topology, not about the GRN's validity.
- The multi-omic data layers and the off-target repair are untouched.

---

## 7. One-line summary

The RAC rebuild was executed substrate-first and stopped at its own gate: a directed motif-regulatory
graph fails to predict PDAC-TF essentiality beyond node degree exactly as the co-expression graph
did (no property > 0.55 matched AUC; all partial correlations n.s.), so the §18 structural failure
is confirmed across two independent topologies — the signal is not in the graph, and no dynamics
over it can recover it.

---

## 8. The achievable ceiling (Phase 4) — and the one weak positive of the whole investigation

The rebuild stopped at the substrate gate (§3): no graph topology beats degree. That leaves the
quantitative question — is degree the ceiling for *any* available feature, or only for topology? A
nested-cross-validated supervised model (logistic + gradient boosting, 5 seeds, in-fold
standardisation, fixed defaults — no hyperparameter search on the labels) was fit on every
leakage-free feature: co-expression degree, all directed-GRN properties, CNA, methylation,
expression mean/variance (normalised and raw), and disease log2FC.
Scripts: `supervised_ceiling.py`, `locate_ceiling_signal.py`, `selective_ceiling.py`,
`selective_null.py`, `selective_robustness.py`.

### 8.1 Absolute essentiality — beats degree, but on a confound

Full model CV-AUC **0.85** vs degree-alone **0.62**. But locating the signal
(`locate_ceiling_signal.py`) shows it is carried by **expression level**, not topology: univariate
CV-AUC `expr_mean_raw` **0.809**, `expr_var` 0.70, every graph property ≤ degree (0.62), CNA ≈ 0.52.
This is the known DepMap tautology: a gene not expressed in a line has Chronos ≈ 0 (knocking out an
untranscribed gene does nothing), so "expressed" is a near-necessary condition for "essential". The
absolute ceiling is real but recovers *being expressed*, which is not a therapeutic target.

### 8.2 Selective essentiality — the endpoint that matters, tested to destruction

PDAC-selective essentiality (Chronos PDAC minus other lineages) is a *difference*, which cancels the
expression-level confound. Degree is **anti**-predictive here (AUC 0.424 — hubs are core-essential,
not selective). The full multi-omic model reaches **0.651**. This was then held to the same bar that
killed the collapse selective hint in §15b:

- **Matched permutation null** (`selective_null.py`, identical 16-feature / 5-seed protocol): observed
  0.651 vs null mean 0.489, p95 0.639, **p = 0.030**. Significant — but marginal, and I first ran the
  null on the wrong (14-feature) set, where it collapsed to 0.516; corrected to match the 0.651
  exactly before trusting it.
- **Threshold robustness** (`selective_robustness.py`, cuts 0.10–0.20, the §15b test):

| selective cut | n positive | observed AUC | null mean | perm p |
|---|---|---|---|---|
| 0.100 | 20 | 0.701 | 0.498 | 0.010 |
| 0.125 | 16 | 0.693 | 0.503 | 0.020 |
| 0.150 | 14 | 0.651 | 0.498 | 0.095 |
| 0.175 | 9 | 0.647 | 0.480 | 0.130 |
| 0.200 | 6 | 0.553 | 0.488 | 0.365 |

**This is not the §15b artifact, and the naive "significant at 2/5 cuts → artifact" auto-verdict is
overruled by reading the pattern.** The §15b collapse hint *bounced* (0.605 / 0.573 / 0.636 / 0.537 /
0.471 — effect size fell below 0.5). Here the observed AUC is **stable at ~0.65–0.70**, and sits
**~0.15–0.20 above the null mean at every cut**, including the non-significant ones (0.651 vs 0.498;
0.647 vs 0.480). Significance is lost at the high cuts only because the positive count drops to 9
then 6, inflating the null's p95 — a **power** effect, not an effect-size collapse. That is the
signature of a real but underpowered signal, and it is qualitatively different from the collapse
hint.

### 8.3 The honest verdict — weak, real, underpowered

> **SUPERSEDED BY §9.** The confound this section names but does not test was tested in §9, and the
> claim does not survive intact: a *single* expression feature (`expr_mean_raw`, univariate CV-AUC
> **0.777**) outperforms this entire 16-feature model (0.651 / 0.683), and removing expression
> collapses the linear model to 0.450. Read §9 before relying on anything below.

A supervised multi-omic model recovers a **weak PDAC-selective essentiality signal that degree,
graph topology, and RAC dynamics all miss** (degree is anti-predictive at 0.424). It is significant
at the best-powered label cuts (p = 0.010, 0.020 at n = 20, 16), with a consistent effect size that
fades to non-significance only as the positive class shrinks below ~14. Caveats that keep it a
hypothesis, not a result: ~20 genes at best; marginal after accounting for the five cuts tested; and
it leans on raw expression, which for the selective difference is biologically plausible but not
proven free of a measurement confound. It should be tested prospectively on a larger PDAC-selective
dependency panel, not reported as an established predictor.

### 8.4 What the rebuild concluded

- **RAC's mechanism (attractor dynamics over a graph) is a dead end for essentiality** — retracted
  (§15), retired (§17), unrescuable by gain (§18) or by a directed substrate (§3 here).
- **The little therapeutically-relevant signal that exists is recoverable by ordinary supervised
  learning over the assembled multi-omic data** — expression, CNA, methylation, disease log2FC — not
  by the network dynamics. The value produced across this project is the *data assembly* and the
  *intervention gate* (§16), not the attractor model.
- The rebuild therefore ends not with RAC v2, but with a clear, evidence-backed statement of what
  works (a weak supervised multi-omic selective signal, underpowered) and what does not (everything
  attractor-shaped), which is the honest and useful outcome.

---

## 9. Phase 5 — the confound test: the last positive is expression, and the model is worse than its own feature

§8.3 conceded the danger it never tested: *"it leans on raw expression, which for the selective
difference is biologically plausible but not proven free of a measurement confound."* This is that
test. Script `scripts/selective_confound_test.py` → `results/selective_confound_test.json`.

**Why the confound is serious.** A gene can only be essential in a cell where it is expressed, and
DepMap expression and DepMap CRISPR are measured in *the same cell lines*. So an expression feature
can predict the PDAC arm of `sel = −(pdac_chronos − other_chronos)` near-definitionally, without any
discovery having occurred. Note what had and had not been measured before: `selective_ceiling.py`'s
artifact check computed univariate AUCs only for ABSOLUTE essentiality (where `expr_mean_raw` alone
scores 0.809 against a full model of 0.847 — absolute essentiality was already conceded in §8.1 to
be essentially an expression detector). **No feature had ever had its univariate AUC measured on the
SELECTIVE endpoint** — the one claim still standing.

**Protocol guard first.** The full 16-feature model reproduces at **0.6510 logistic** (expected
0.651) and **0.6833 GBM** (expected 0.683) on the identical CV protocol and gene set. Every number
below is therefore directly comparable. (This guard exists because I previously ran a null on a
mismatched 14-feature set and got a spurious collapse.)

### 9.1 A single expression feature beats the entire multi-omic model

Univariate CV-AUC on the **selective** endpoint, all 16 features plus the new differential:

| feature | CV-AUC | |
|---|---|---|
| **expr_mean_raw** | **0.7768** | ← expression |
| disease_log2fc_LEVEL | 0.6103 | ← expression |
| expr_var_norm | 0.6035 | ← expression |
| expr_mean_norm | 0.5994 | ← expression |
| in_strength | 0.5971 | |
| cna_amp | 0.5663 | |
| expr_pdac_minus_other_DIFFERENTIAL | 0.5566 | ← expression (new) |
| methylation | 0.5557 | |
| cna_mean | 0.5437 | |
| out_strength | 0.5296 | |
| eigenvector | 0.5029 | |
| in_degree | 0.4526 | |
| coexpr_degree | 0.4242 | |
| expr_var_raw | 0.4116 | ← expression |
| hits_hub | 0.3795 | |
| out_degree | 0.3760 | |
| pagerank | 0.3630 | |

**`expr_mean_raw` alone (0.777) outperforms the full 16-feature model (0.651 logistic, 0.683 GBM).**
The multi-omic integration does not add value to this endpoint — it *dilutes* a single strong
feature. The §8 framing ("a supervised multi-omic model recovers selective signal that degree
misses") therefore cannot stand as written: what recovers the signal is mean PDAC expression, and
the model is worse than that one column.

### 9.2 My predeclared hypothesis was wrong — and the correction is informative

I predeclared that the **PDAC-vs-other expression differential** would carry the confound
(threshold: ≥ 0.62 ⇒ "tautology-dominated"), reasoning that "selectively expressed ⇒ selectively
essential" was the tautology. **That is refuted.** The differential scores only **0.5566**, and
adding it to the full model *hurts* (0.651 → 0.634).

The confound is real but takes a different form than I predicted. It is **absolute PDAC expression
level**, not the PDAC-vs-other contrast. The mechanism: `expr_mean_raw` predicts absolute
essentiality at 0.809 and selective at 0.777 — nearly equally. Since the selective endpoint is a
difference whose PDAC term is itself expression-driven, high PDAC expression inflates the PDAC arm
of the contrast. Expression is confounding the *minuend*, not the difference. Recording the
refutation of my own hypothesis matters: the naive tautology I expected is not what is happening,
and a reader checking only the differential would have wrongly cleared the finding.

### 9.3 Ablation splits by model class — reported, not collapsed

> **PARTIALLY RETRACTED BY §10.** The "inverted predictiveness is predictiveness … selective
> dependencies sit at the periphery of the co-expression graph" inference below is **unsound and
> withdrawn**. Fitted logistic OOF AUC is exactly invariant to negating a feature, and at 14
> positives it drags genuinely-predictive features below 0.5, so the sub-0.5 entries carry no
> directional information. Model-free, the centrality features mostly sit slightly *above* 0.5.
> The ablation numbers themselves stand; the peripherality interpretation does not. See §10.

Removing all five expression-derived features (leaving 11: topology + CNA + methylation):

| model | with expression | without expression |
|---|---|---|
| logistic | 0.651 ± 0.010 | **0.450 ± 0.027** |
| GBM | 0.683 ± 0.031 | **0.625 ± 0.043** |

The script's auto-verdict read only the logistic and returned "EXPRESSION-CARRIED". That is too
clean, and the honest statement is the split: **the linear signal is entirely expression** (0.450 is
below chance), while **a nonlinear model retains 0.625** on topology + CNA + methylation alone.

That GBM residual is a genuine observation and deserves its caveats stated with it: 14 positives
across 5 folds is ~2.8 positives per fold, the seed spread is the widest in the table (±0.043), and
GBM has the most freedom to fit noise. What it is likely learning is visible in the univariate
table — several topology features are strongly **anti**-predictive (pagerank 0.363, out_degree
0.376, hits_hub 0.380, coexpr_degree 0.424). Inverted predictiveness is predictiveness: *selective
dependencies sit at the periphery of the co-expression graph, not at hubs*. That is consistent with
§15b, where collapse (a propagation/centrality-like score) ranked KRAS — the strongest selective
dependency — at the 8th percentile. It is reported as an **underpowered hypothesis**, not a result.

**A limitation of the ablation, disclosed.** Removing expression *levels* does not remove all
expression-derived information: `coexpr_degree` and `eigenvector` are built from the co-expression
graph, which is itself derived from expression correlations. This is a partial ablation. Mitigating
it: §15 established topology is anti-predictive for selective essentiality (0.424), so topology is
not smuggling in the expression signal — if anything it carries inverted information.

### 9.4 Corrected statement of the investigation's one positive

The §8.3 claim is re-described, not deleted:

- **Not supported:** "a supervised *multi-omic* model recovers a weak PDAC-selective essentiality
  signal." The multi-omic model is *worse* than one of its own features.
- **Supported, with a heavy confound caveat:** PDAC-selective essentiality is predicted by **mean
  PDAC expression** (univariate CV-AUC 0.777), a channel that is substantially near-definitional —
  expression and CRISPR are measured in the same lines and expression drives the PDAC arm of the
  essentiality contrast. This is expected biology, not a discovery, and it needs neither a network
  model nor multi-omic integration to state.
- **Hypothesis only:** a nonlinear residual (GBM 0.625 without expression) suggests selective
  dependencies are graph-*peripheral* rather than hubs — 14 positives, widest variance in the study,
  needs a larger selective panel to test.

**Net effect on the whole investigation.** Every attractor-shaped claim was already retracted
(§15), retired (§17), and shown unrescuable (§18, RAC v2 §3). The single positive that survived to
Phase 4 now reduces largely to an expression confound. What remains genuinely defensible from this
project is the **data assembly** and the **data-calibrated intervention gate** (§16, 11/12 roles
CNA-corroborated) — not the attractor model, and not the supervised ceiling.

---

## 10. Phase 6 — the graph-peripheral hypothesis is refuted, and the screen that produced it was unsound

§9.3 offered one new observation: on the selective endpoint several centrality features scored below
0.5 (pagerank 0.363, out_degree 0.376, hits_hub 0.380, coexpr_degree 0.424), from which I inferred
*"inverted predictiveness is predictiveness: selective dependencies sit at the periphery of the
co-expression graph."* Tested, that inference is wrong twice over — once in the hypothesis and once
in the instrument that produced it. Scripts: `peripherality_test.py`, `univariate_screen_modelfree.py`.

### 10.1 Two demonstrated errors in the Phase 5 screen

**(a) Logistic CV-AUC is exactly invariant to negating a feature.** Fitting on `−x` flips the learned
coefficient and leaves the predicted probabilities identical. Verified to machine precision:
`AUC(x) = AUC(−x) = 0.423986`. The Phase 6 "peripherality = −centrality" composite was therefore a
**no-op** — it re-tested centrality under a new name. Any construction of the form "invert the
feature and re-score with a model that learns its own sign" is meaningless.

**(b) At 14 positives, a fitted OOF AUC below 0.5 does not mean inverted signal — it means the
relationship does not generalise.** Verified on synthetic data built to have a genuinely *positive*
association (`x = noise + 0.4·y`, model-free AUC **0.562**): the 5-fold logistic OOF AUC comes out at
**0.424**, below chance, purely from instability at ~2.8 positives per test fold. A real effect is
dragged below 0.5 by the protocol itself.

Together these mean the sub-0.5 rows of the §9 table carry **no directional information**, and the
peripherality claim built on them is withdrawn.

### 10.2 The model-free screen — several directions flip

Recomputing univariately with **rank AUC** (no fitting, no folds, so no sign-learning and no
small-fold instability) plus Spearman against the *continuous* selective endpoint (all 419 genes,
far better powered than a 14-positive dichotomy):

| feature | fitted OOF (§9) | model-free rank AUC | direction | ρ (continuous) | p | q (BH) |
|---|---|---|---|---|---|---|
| expr_mean_raw | 0.777 | **0.794** | higher ⇒ more selective | 0.111 | 0.024 | 0.402 |
| expr_var_norm | 0.604 | **0.373** | *flipped* | −0.064 | 0.194 | 0.474 |
| expr_mean_norm | 0.599 | 0.627 | — | 0.005 | 0.920 | 0.956 |
| in_strength | 0.597 | **0.378** | *flipped* | −0.016 | 0.741 | 0.940 |
| pagerank | 0.363 | 0.386 | higher ⇒ less selective | 0.070 | 0.152 | 0.474 |
| expr_pdac_minus_other_DIFF | 0.557 | 0.611 | — | 0.083 | 0.091 | 0.474 |
| out_degree | 0.376 | **0.558** | *flipped* | 0.015 | 0.766 | 0.940 |
| hits_hub | 0.380 | **0.556** | *flipped* | 0.014 | 0.774 | 0.940 |
| expr_var_raw | 0.412 | **0.549** | *flipped* | 0.035 | 0.481 | 0.743 |
| coexpr_degree | 0.424 | **0.527** | *flipped* | 0.063 | 0.195 | 0.474 |

**Six features flip direction.** The centralities that founded the peripherality hypothesis
(out_degree, hits_hub, coexpr_degree) move from "anti-predictive" to *slightly above* 0.5 — the
opposite of peripheral — while pagerank and in_strength stay below. Directions are inconsistent
among centralities, which is the signature of noise, not structure.

### 10.3 Verdict, and what it does to the rest of the record

**The graph-peripheral hypothesis is REFUTED.** No centrality feature survives Benjamini-Hochberg;
in fact **nothing does** — over 17 continuous tests the best raw p is expr_mean_raw at 0.024, giving
**q = 0.402**. At this sample size the selective endpoint cannot support *any* feature-level claim
that survives multiple-testing correction.

Note the nuance this exposes in the surviving expression result: `expr_mean_raw` is strong on the
**dichotomised tail** (rank AUC 0.794) but only weakly monotone across all genes (ρ = 0.111). It
separates the extreme selective genes rather than tracking selectivity continuously — consistent
with §9's confound account (the top-selective genes are highly expressed), and a further reason not
to promote it to a predictor.

**Transferable methodological lesson**, worth more than the hypothesis it killed: *for univariate
screening with a small positive class, use a model-free rank statistic.* A fitted cross-validated
model (i) cannot express direction, because it learns its own sign, and (ii) is biased below chance
by fold instability. Reading direction off fitted OOF AUCs — which is what §9.3 did — is unsound,
and it produced a confident, wrong, biologically-plausible story about network topology.

---

## 11. Phase 7 — the detection floor: what this design could and could not have seen

§10 ended on "nothing survives BH", which is an ambiguous stopping point: it is consistent with
*there is no signal* and with *we could never have seen one*, and those demand opposite follow-ups.
This quantifies the floor. Script: `scripts/selective_power_floor.py` →
`results/selective_power_floor.json`. Method: the univariate screen is a rank comparison of 14
positives against 405 negatives, so its power is Mann-Whitney power; for a target rank-AUC `a`,
normal scores separated by `d = √2·Φ⁻¹(a)` reproduce that AUC exactly. BH across 17 features is
bounded conservatively by Bonferroni α/17 (correct for the best-ranked feature).

### 11.1 Power at the actual design

BH-corrected power by design size and true effect:

| n positive | AUC 0.60 | 0.65 | 0.70 | 0.75 | 0.80 |
|---|---|---|---|---|---|
| **14 (actual)** | 0.07 | 0.19 | 0.41 | 0.70 | 0.91 |
| 20 | 0.10 | 0.31 | 0.62 | 0.89 | 0.98 |
| 30 | 0.17 | 0.49 | 0.84 | 0.98 | 1.00 |
| 50 | 0.32 | 0.77 | 0.98 | 1.00 | 1.00 |
| 100 | 0.61 | 0.97 | 1.00 | 1.00 | 1.00 |

**Minimum detectable rank-AUC at 80% power, 14 positives: 0.771 (BH), 0.691 (nominal).** Positives
required for the effects actually observed: `expr_mean_raw` (0.794) → **14, i.e. we had enough**;
the expression differential (0.611) → **120**; the best centrality, eigenvector (0.563) → **>400**;
`coexpr_degree` (0.527) → **>400**.

### 11.2 A correction to §10: "nothing survives BH" was over-generalised

§10 applied BH to the **continuous** Spearman tests and concluded that the selective endpoint
"supports no feature-level claim that survives multiple testing". That is true of the continuous
test and **false of the dichotomised one**. Recomputing BH over Mann-Whitney p-values on the
dichotomised endpoint (normal approximation from the rank AUCs, ample at 14 vs 405):

| feature | p (dichotomised MWU) | q (BH) | survives? |
|---|---|---|---|
| **expr_mean_raw** | **0.00018** | **0.0030** | **YES** |
| expr_var_norm | 0.105 | 0.374 | no |
| expr_mean_norm | 0.105 | 0.374 | no |
| disease_log2fc_LEVEL | 0.105 | 0.374 | no |
| in_strength | 0.121 | 0.374 | no |

**`expr_mean_raw` survives multiple-testing correction robustly (q = 0.0030); nothing else does.**
(The first attempt at this table had a BH bug of my own — a forward running minimum instead of the
reverse cumulative minimum BH's step-up requires — which reported q = 0.003 for a p of 0.105. Caught
because a q three orders below its own p is impossible; corrected above.)

### 11.3 The synthesis — a threshold confound, not a graded predictor

Putting Phases 5–7 together, the picture is consistent and sharper than any single phase:

- **Expression separates the extreme selective tail, robustly and detectably.** Rank AUC 0.794,
  MWU p = 0.00018, BH q = 0.0030, and the power analysis confirms 14 positives suffice for this
  effect size. This is real.
- **Expression does not grade selectivity.** Continuous Spearman ρ = 0.111, q = 0.402 — it fails
  correction across the full range.
- **That combination is the signature of a threshold-like confound**, not a biological predictor:
  *whether a gene is expressed in PDAC at all* cleanly separates the top-14 selective genes while
  carrying almost no monotone information across the range — exactly the near-definitional channel
  §9/§21 identified (a gene cannot be essential where it is not expressed, and expression and CRISPR
  are measured in the same lines).
- **Everything else was undetectable by construction.** The differential needed ~120 positives and
  the centralities >400; at 14, power for a 0.65 effect is **7%**. Their non-significance carries no
  information.

### 11.4 Calibrating the §10 refutation — the null was not powered, the directions were

This forces a correction to my own language. §10 called the graph-peripheral hypothesis "REFUTED".
With power of 7–19% for effects in the 0.60–0.65 range, **non-significance alone proves nothing**
about centrality effects of that size in either direction. The refutation therefore rests entirely
on its *other* leg — that the model-free point estimates **contradict** the hypothesis (centralities
landed slightly *above* 0.5, opposite to "peripheral", and disagreed with one another) — not on the
null. The calibrated verdict: **unsupported, with point estimates pointing the wrong way, and
underpowered to settle definitively.** That is weaker than "refuted" and it is what the data
supports.

### 11.5 What a future study needs

The floor converts this null into a design specification. To detect, at 80% power with BH across
~17 features:

| target effect | positives required |
|---|---|
| rank-AUC 0.75 | ~20 |
| rank-AUC 0.70 | ~30 |
| rank-AUC 0.65 | ~50–60 |
| rank-AUC 0.60 | ~100+ |

The panel here has **14**. The recurring §15b/single-cell finding — that only ~7 TFs in 1,164 are
strongly PDAC-selective — means this is not fixable by better statistics on DepMap; it needs a
larger or differently-constituted selective-dependency panel. That is the same information gap
§5 identified, now with a number attached.
