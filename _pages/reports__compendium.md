---
layout: default
title: "Compendium"
subtitle: "A single compiled record of what was built, every data source, method, result, and retraction."
description: "A single compiled record of what was built, every data source, method, result, and retraction."
permalink: /reports/compendium/
group: reports
order: 1
---

A single, in-depth record of the entire project: what was built, every data source, every method,
every result, every retraction, and the one claim that survived. It supersedes nothing and
duplicates nothing — it *compiles* the material spread across
README.md, [METHODS.md]({{ '/reports/methods-full/' | relative_url }}), [FINDINGS.md]({{ '/reports/findings/' | relative_url }}),
[RESULTS.md]({{ '/reports/results-ledger/' | relative_url }}), [REVIEW_RESPONSE.md]({{ '/reports/review/' | relative_url }}) (§1–§28), and the three addenda
([ADDENDUM_DYNAMICS.md]({{ '/addenda/dynamics/' | relative_url }}),
[ADDENDUM_RAC_V2.md]({{ '/addenda/rac-v2/' | relative_url }}),
[ADDENDUM_CHROMATIN.md]({{ '/addenda/chromatin/' | relative_url }})).

Every number below is read from a committed artifact in `results/` or a manifest in
`data/manifests/`; where a claim was later overturned, the overturn is stated in place, not hidden.

---

## 0. One-paragraph summary

`pdac-circuit` is a Python-only, from-scratch pipeline that designs synthetic gene circuits to
target pancreatic-ductal-adenocarcinoma (PDAC) transcription factors, built on ~112 GB of real open
data. Its original headline — a novel "Regulatory Attractor Control" (RAC) model whose
attractor-collapse score predicts CRISPR essentiality — **did not survive** a review-driven
head-to-head test and is **retracted**; a deep audit further showed the model is **not bistable at
all**, so the mechanism it claimed does not exist. Across a 28-step review arc almost every
substantive claim was retracted, retired, shown unrescuable, or exposed as confounded. **Exactly one
positive survived** every attempt to kill it: the 20 RAC-surfaced genes sit on PDAC-gained promoter
H3K27ac beyond expression, beyond their selection variable, and beyond hub-ness — a *real,
promoter-local, modest* effect (~1.5–1.8×), significant across 12/12 parameter settings, though ATAC
does not replicate it. The project's durable value is the **multi-omic data assembly**, the
**data-calibrated intervention gate**, and a body of **methodology** for not fooling yourself.

---

## 1. What was built — the seven-module pipeline

A from-scratch design pipeline. No external pretrained model supplies candidate features,
pseudo-labels, or training targets; frozen Enformer/Borzoi predictions are permitted only as
hash-locked evaluation baselines.

| # | module (`src/pdac_circuit/…`) | function |
|---|---|---|
| I | `targeting/` | rank PDAC-driver TFs (TCGA-PAAD vs GTEx, IntOGen/NCG, Moffitt subtype signatures, MCDA) |
| II | `parts/` | **trained** promoter-strength (RF+CNN) and enhancer-activity (CNN) models + CRISPRi repressor selection |
| III | `circuit/` | AND/NOT logic gates + feedback; Boolean-network + Hill-ODE viability/robustness |
| IV | `seqopt/` | GC, cryptic-splice, restriction sites, codon (CAI Viterbi), 5′ structure |
| V | `grna/` | PAM scan, **trained** on-target (GBT+CNN), CFD/MIT off-target |
| VI | `scoring/` | efficacy/specificity/robustness/safety → NSGA-II Pareto |
| VII | `generate/` | **trained** promoter WGAN-GP generating novel synthetic promoters |
| VIII | `attractor/` | **RAC** — the novel contribution; see §3 (and its retraction) |

### 1.1 Trained models (real held-out data, from scratch)

| model | module | architecture | held-out metric |
|---|---|---|---|
| promoter strength | II | RF + CNN | Spearman **0.5275** (full FANTOM5 209k real peaks; fixed chr8/9 held-out) |
| enhancer activity | II | CNN | AUROC **0.815** (full uncapped pancreas 20k→80k actives; generalizes to PANC-1 PDAC at 0.835) |
| gRNA on-target | V | GBT + CNN | Spearman **0.657** (Doench-2016 + Kim-2019, 18,142 guides; CNN + GBM both trained on the merged set) |
| promoter generator | VII | WGAN-GP | retrained on full 52k real promoters (was 12k); certified-real (4-mer JS **0.012** ≪ 0.05, beats random) with stronger selectable tail p90 0.94→**0.99** |
| long-range chromatin (PDACircuitFormer) | — | 196,608 bp → 1,536 bins, 2.25 M params | held-out profile r **0.7102 ± 0.0091** (8 seeds) — *optimisation stability only* |

---

## 2. Data — 112 GB, 381 artifacts, all REAL, sha256-verified

Every corpus is downloaded, hashed, and recorded under `data/manifests/` with `dataClass: REAL`;
`sha256` is never fabricated; license-walled sets (COSMIC/OncoKB) are replaced with open
equivalents rather than bypassed.

| corpus | contents | role |
|---|---|---|
| `hg38-ref` | GRCh38 reference (0.98 GB) | genome for PWM scan, off-target search |
| `depmap-crispr` | Chronos gene effect, 1,164 TFs × 1,684 lines (0.44 GB) | essentiality readout (held out of every fit) |
| DepMap expression | `OmicsExpression.csv` (hardlinked) | co-expression graph, `expr_mean_raw` |
| `tcga-paad` | RSEM expression, GISTIC CNA, HM450 methylation, RPPA | tumour multi-omics |
| `gtex-pancreas` | GTEx v8 median TPM | normal-pancreas expression reference |
| `cptac-pdac` | 238 tumours × 12,017 proteins (0.04 GB) | protein layer (651/1,639 Lambert TFs) |
| `tisch-paad` | Peng-2019 scRNA, 57,443 cells (11,401 malignant) | in-vivo malignant graph |
| `4dn-panc1-hic` | PANC-1 A/B compartments, insulation, TAD boundaries | 3-D genome |
| `encode-bulk` | 330 files, **101.85 GB** healthy-pancreas ChIP/ATAC BAMs+bigWigs | healthy chromatin prior |
| `encode-panc1-pdac` | 11 Panc1 signal-pval tracks (0.90 GB) | PDAC cell-line chromatin |
| `encode-foldchange` | **4 fold-change-over-control** bigWigs (3.55 GB) | §25 chromatin recompute |
| `fantom5-cage` | CAGE promoters (0.84 GB) | promoter-model training |
| `gencode-v46` | gene models (0.05 GB) | TSS coordinates |
| `dbsnp-common` | common SNPs (1.59 GB) | guide SNP-overlap flag |
| `doench2016-cfd` | exact CFD nucleotide-pair matrix (plain text) | §24 off-target scoring |
| `lambert-tf`, `intogen-pdac`, `ncg` | TF catalogue, PDAC drivers, cancer-gene census | target universe |

**Provenance gotchas recorded:** `load_tcga_paad_expression(genes)` silently re-fetches and
overwrites the cache (call it with no args); ENCODE 405s a Mozilla UA (use a project UA); 4DN
`@@download` 403s (use `open_data_url`); the exact CFD matrix exists as **plain text** in
crisprScore, so CRISPOR's `.pkl` (arbitrary code execution) was never used.

---

## 3. RAC — the novel contribution, and the arc that dismantled it

### 3.1 What RAC was

A **bistable graph dynamical system** `x ← σ(gain·(Wx + b))`, gain = 4.0, with `W` masked to a
DepMap co-expression graph and sign-anchored to correlations. The 54 real PDAC cell states were fit
as the "viable" attractor; a low state was penalised to stay "dead". **Essentiality** was defined as
the network-wide **collapse toward the dead attractor** when a node is clamped down. Targets are the
genes whose clamping best collapses the system while an intervention gate keeps the direction sane.
The 20 surfaced targets: BRCA2, GATA6, ZNF790, SETDB1, KMT2C, E2F1, SOX13, AHR, MYBL2, AGR2, ZNF331,
SF3B1, SMAD3, HOXA3, ATM, ZNF528, FAM83A, ZNF93, FOSL1, ZNF85.

### 3.2 The original claim, and its retraction (§1, §15, §15b)

**Claimed:** collapse predicts held-out CRISPR essentiality, AUC 0.653 (50,000-permutation
p = 0.0022), beating degree (0.629) and eigenvector centrality (0.584).

**The test that had never been run** — a direct head-to-head against the degree baseline
(`rigorous_validation.py`):

| statistic | value |
|---|---|
| AUC collapse vs degree | 0.547 vs **0.629** |
| ΔAUC (paired bootstrap) | **−0.082**, 95% CI [−0.199, +0.029] |
| partial ρ given degree/expr/variance | **0.028, p = 0.56** |
| CV covariates vs +collapse | 0.653 → 0.652 |

**Collapse adds nothing beyond degree.** The permutation had tested against *chance*, not against
degree, and the configuration had been selected on the same CRISPR labels. **Retracted.**

**Located (§15):** a degree-conditioned analysis showed collapse *is* degree in disguise for core
essentiality (degree-matched AUC ≤ 0.49). **Dissected (§15b):** the one flattering exception, a
PDAC-selective hint, bounced across label cuts and had only 1/14 positives in the informative
quadrant — collapse ranks **KRAS**, the strongest selective dependency, at the **8th percentile**.
Artifact.

### 3.3 The foundational finding: the system is not bistable (§17, ADDENDUM_DYNAMICS)

The entire method rests on bistability — a contractive/monostable map has one fixed point and cannot
express "collapse". This had never been checked. Four tests, three of them model-free:

| test | result | interpretation |
|---|---|---|
| cell-state one-step residual | median **0.73**, max 0.98 | the "attractors" are not fixed points |
| convergence from 84 inits (2,000 iters) | **0 / 84** converge | no reachable stable attractor |
| perturbation growth (20 steps) | median **3.4×**, max 71× | dynamics are expansive/unstable |
| spectral radius ρ | **1.02–1.13** (analytic = finite-diff) | linearly unstable |
| bifurcation (gain sweep) | hysteresis only from gain **5.9** | operating gain 4.0 is below it |
| clamping reaches a dead basin | **0 / 10** nodes | there is no dead basin to reach |

`collapse_scores` iterates a *non-converging* map for 250 steps and compares two transient
snapshots — a **graph-influence propagation** score, not a basin transition, which is *why* it
equals degree. **The bistable-attractor framing is retired.**

*(A self-caught bug: the first spectral radius was ρ≈2.6, from evaluating σ′ at `z` instead of the
map's argument `gain·z`. A finite-difference cross-check, placed specifically to catch that class of
error, flagged it. The conclusion never depended on the eigenvalue — the convergence and
perturbation tests are model-free.)*

### 3.4 Not rescuable (§18, §19)

- **Raising the gain (§18):** refit at gains 4–8 → **0.00 convergence at every gain**; collapse never
  robustly beats degree. The bistable regime is never reached and the failure is *structural*.
- **A better substrate (§19, ADDENDUM_RAC_V2):** a directed TF→target motif-GRN (`build_directed_grn.py`,
  422 nodes, 205 regulators, 11,331 edges, leakage-free vs CRISPR) — **no** directed property beats
  degree (out-strength 0.39, out-degree 0.43, pagerank 0.48, authority 0.54; all matched-AUC < 0.55).
  Two independent topologies both hit the degree ceiling: essentiality is **not in the graph
  topology** beyond node degree.

### 3.5 The achievable ceiling, and the last confound (§20–§23)

A supervised nested-CV model on all leakage-free features:

- **Absolute essentiality:** full model 0.85 vs degree 0.62 — but `expr_mean_raw` alone scores 0.809,
  so this is the DepMap expression tautology (a gene can't be essential where it isn't expressed).
  Real, not actionable.
- **PDAC-selective essentiality** (the endpoint that matters; degree anti-predictive at 0.42): full
  model 0.651. §21 confound test: **a single feature, mean PDAC expression, scores 0.777 —
  beating the full 16-feature model**; ablating expression collapses the linear model to 0.450. The
  channel is near-definitional. §22: the "graph-peripheral" follow-up hypothesis was **refuted**, and
  the fitted-CV screen that produced it was **unsound** (a fitted CV-AUC cannot express feature
  direction, and at 14 positives is biased below chance — model-free re-screening flipped six
  directions; nothing survives BH). §23 detection floor: at 14 positives the minimum detectable
  rank-AUC is 0.771, so most effects were undetectable by construction — a design specification, not
  a discovery.

**Net:** no regime — core or selective, expression-blind or not, linear or nonlinear — recovers
essentiality signal that degree misses. RAC as a predictor is dead, on the merits, in every form
tested.

---

## 4. What survives, and how hard it was tested — the H3K27ac result (§25–§28, ADDENDUM_CHROMATIN)

The one claim that survived every attempt to kill it.

**Claim:** the 20 RAC-surfaced genes sit on **PDAC-gained H3K27ac at their promoters** (Panc1 vs
healthy pancreas), not explained by expression, by the disease-expression change they were selected
on, or by hub-ness.

**Why the original number was wrong:** it used ENCODE **signal p-value** tracks (depth-confounded)
and an **unmatched** background (targets are high-expression hubs, so the contrast risked measuring
expression). Both fixed: fold-change-over-control tracks matched to the *same processing run*
(`derived_from`) as the p-value tracks they replace, plus three matched backgrounds.

### 4.1 Matched controls (§25)

| contrast | matched score | targets | background | MWU p |
|---|---|---|---|---|
| all background | — | +0.919 | −0.091 | **0.0022** |
| absolute expression | 2.232 vs 2.230 | +0.919 | +0.105 | **0.017** |
| `disease_log2fc` (circularity) | 6.867 vs 6.75 | +0.919 | +0.022 | **0.025** |
| co-expression degree (hub-ness) | 136.8 vs 136.75 | +0.919 | −0.017 | **0.010** |

### 4.2 Fragility, window, pseudocount (§26–§28) — 12/12 settings significant

| axis | range | significant | notes |
|---|---|---|---|
| set-level permutation | 20k draws of 20 genes | **p = 0.00155** | the correct null; *stricter* than gene-wise MWU |
| leave-one-target-out | all 20 | worst **p = 0.0058** (drop HOXA3) | not one gene |
| caliper | 0.10 / 0.25 / 0.50 SD | 3/3 | flat |
| window | ±500 bp – ±25 kb | **6/6** | promoter-local (see below) |
| pseudocount | 0.01 – 2.0 | **6/6** | MWU p 0.0021–0.0032 (rank test → constant-immune) |

### 4.3 Two corrections the aggregate was hiding

- **Promoter-local, not domain-wide:** absolute target enrichment falls from 2.05× (±500 bp) to
  **1.01× (±25 kb)** — no gain at domain scale; wide-window significance comes from *background*
  depletion. Not a "PDAC-gained enhancer domain".
- **The top locus is a pseudocount artifact:** HOXA3's healthy fold-change is exactly 0.000, so its
  58.7× ratio is set by `PSEUDO = 0.1`, not data. It's the only target with near-zero healthy signal.

### 4.4 An unplanned coherence check that passes

**GATA6 is the most negative target (0.21×).** GATA6 is the classical-identity factor and §9 showed
PANC-1 sits below the panel mean on the classical programme — a classical enhancer should be *less*
acetylated in a non-classical line, and it is. Nobody designed this check.

### 4.5 Effect size — a range, because it is not well-determined

| estimator | published settings | span over 12 settings |
|---|---|---|
| raw mean | 1.89× | 1.45×–2.15× |
| **median** | **1.81×** | 1.48×–1.85× |
| **mean excl. zero-denominator** | **1.58×** | 1.37×–1.60× |

**Existence robust; magnitude ~1.5–1.8×, plausibly 1.0–2.2×.** No single headline number is
supported. **ATAC does not replicate** (p = 0.074), so this is H3K27ac-specific. It does **not**
resurrect RAC — it says the *gene set* has a chromatin property.

---

## 5. The intervention gate — data-calibrated (§4, §4b, §16)

The convergence score ranked genes by attractor movement and never asked whether repressing them was
biologically sane; it passed TGIF1, whose *loss* accelerates PDAC. The **signed intervention gate**
(`attractor/intervention_gate.py`) assigns each gene a curated role and admissible direction, and
blocks/quarantines the rest. Only **4 of 20** ranked targets have a defensible repression direction.

**§16 calibration** against independent TCGA copy number: **11/12 directional role calls
corroborated** — all seven repression-allowed oncogenes are amplified; four of five tumour-suppressor
blocks are deleted. The sole conflict, **GATA6** (amplified yet quarantined), is the gate *working*:
an amplified dependency that is still unsafe to repress because its loss drives the aggressive basal
state — exactly the case a dependency ranking gets wrong. *(Gotcha found here: `RegulatoryGraph.disease_log2fc`
is a cross-platform TCGA-vs-GTEx differential dominated by a uniform batch offset, useless as a
direction; CNA is the only clean bidirectional signal locally.)*

---

## 6. The four designed constructs — and why none is orderable (§13, §14, §24)

Modules I–VI were run end-to-end on the four gate-approved oncogenic targets, producing real
protospacer+PAM guides in open chromatin, no common-SNP overlap, each **uniquely placed** (exactly
one perfect genomic match). But the shipped off-target `risk = 0.00` was **blind, not clean**: the
search covered the target loci ±5 kb — **0.0013% of the genome**. A real hg38 scan (both strands,
≤4 mismatches) finds 35–206 near-matches per guide.

Resolved with the **exact Doench-2016 CFD matrix** (§24, downloaded as plain text, validated against
7 published test vectors, sha256'd): **all four guides fail** the pre-registered specificity gate.

| gene | protospacer+PAM | cfd_specificity ≤4mm | gate |
|---|---|---|---|
| SETDB1 | `ACCCCAGACTCACAACTCAG`+GGG | 0.045 | FAIL |
| MYBL2 | `CGCTGGTGAGACGAGCCGGG`+AGG | 0.082 | FAIL |
| E2F1 | `GGAGATGATGACGATCTGCG`+AGG | 0.126 | FAIL |
| FOSL1 | `TCTGACTCACCCGCGCCGTG`+CGG | 0.308 | FAIL |

*(A prediction of mine was wrong here: I expected the position-granular proxy to be "pessimistic";
it erred both ways and was **optimistic** at ≤3 mm, flipping MYBL2 and E2F1 from pass to fail.)*
**No construct in this repository is ready to order.**

---

## 7. Complete claim ledger

| # | claim | verdict | §/addendum |
|---|---|---|---|
| 1 | collapse predicts CRISPR essentiality | **RETRACTED** (no gain over degree) | §1/§15/§15b |
| 2 | bistable attractor formulation | **RETIRED** (not bistable at any gain) | §17/§18 |
| 3 | RAC rescuable by better substrate | **NO** (directed GRN hits degree ceiling) | §19 |
| 4 | supervised selective ceiling | **EXPRESSION-CARRIED** (model < one feature) | §20/§21 |
| 5 | graph-peripheral hypothesis | **REFUTED** (+ unsound screen) | §22 |
| 6 | leave-cell-line-out generalisation | **DEMOTED** (leaked whole-panel stats) | §3 |
| 7 | off_target_risk = 0.00 | **RETRACTED → resolved: all 4 fail** | §13/§14/§24 |
| 8 | chromatin "3× on two marks" | **SUPERSEDED**: H3K27ac holds, ATAC doesn't | §25–§28 |
| 9 | circuit robustness = 1.00 | **UNINFORMATIVE** (dead circuit scores 1.0) | §11 |
| 10 | promoter "strength" absolute value | **UNINFORMATIVE** (random DNA 0.758) | §12 |
| 11 | basal vs classical target lists | **PREMISE FAILS** (candidates pan-PDAC) | §9 |
| 12 | Module I driver-recovery p = 0.003 | **CORRECTED, survives** (p = 0.013) | §10 |
| 13 | promoter WGAN novelty | **UPHELD** (zero near-exact copies) | §12 |
| 14 | guides uniquely placed | **UPHELD** (one perfect match each) | §14 |
| 15 | intervention gate role calls | **DATA-CALIBRATED** (11/12 via CNA) | §16 |
| 16 | **H3K27ac promoter enrichment** | **SURVIVES** (12/12 settings; ~1.5–1.8×) | §25–§28 |

---

## 8. Methodology — the transferable lessons

The recurring failure across this project was not a wrong number but **a check structurally unable
to detect the failure it existed to catch**. Instances, several of them self-inflicted and caught by
guards built for the purpose:

- an off-target search covering 0.0013% of the genome reporting "0 off-targets";
- a robustness metric a **dead circuit** scores 1.000 on;
- a GAN novelty test (JS divergence) that memorisation *minimises*;
- a fitted CV-AUC used to read feature **direction** — it cannot (AUC(x) = AUC(−x)), and at small n
  is biased below chance ([[gotcha_fitted_cv_auc_direction]]);
- an analytic Jacobian evaluated at the wrong argument (`z` vs `gain·z`), caught by a
  finite-difference cross-check;
- a verdict function that reported "SURVIVES" for ATAC while its **primary** contrast had failed;
- a permutation null with `null >= NaN` always False, giving a spurious p = 0.0003.

Durable practices that came out of it: **model-free rank statistics** for small-positive-class
screens; **matched backgrounds** on the *selection variable*, not just on level; **set-level nulls**
for set-level claims; **parameter sweeps** (window, pseudocount, caliper) to distinguish a real
effect from a single-threshold artifact; **predeclared** interpretation thresholds; and reporting an
**effect-size range** when constants no data constrains move the magnitude.

---

## 9. Honest standing, and what would confirm the survivor

**Durable value of the project:** the 112 GB multi-omic data assembly (real, hashed, governed); the
data-calibrated intervention gate; and the methodology above. **Not** the attractor model, and
**not** any orderable construct.

**To confirm the one surviving result**, a future study needs: (1) a properly-typed substrate —
primary PDAC tumour/organoid H3K27ac, or lines at the poles of the Moffitt signature, not the
intermediate PANC-1; (2) multiple healthy references, not one track; (3) a second mark that
replicates (ATAC did not); (4) an independent target set, since these 20 came from a retracted model.

**Still blocked at time of writing:** the leakage-free leave-cell-line-out rerun, which needs the
GPU (pinned by unrelated processes).

---

## 10. Reproducibility index

Every result maps to a committed script and artifact:

| topic | script(s) | artifact(s) |
|---|---|---|
| collapse vs degree | `rigorous_validation.py` | `rigorous_validation.json` |
| collapse located/dissected | `conditional_collapse_signal.py`, `selective_hint_dissection.py` | `conditional_collapse_signal.json`, `selective_hint_dissection.json` |
| dynamics / bistability | `dynamics_characterization.py`, `verify_dynamics_instability.py`, `gain_sweep_rescue.py` | `dynamics_characterization.json`, `verify_dynamics_instability.json`, `gain_sweep_rescue.json` |
| substrate rebuild | `build_directed_grn.py`, `substrate_signal_test.py` | `directed_grn.npz`, `substrate_signal_test.json` |
| supervised ceiling / confound | `selective_ceiling.py`, `selective_confound_test.py`, `univariate_screen_modelfree.py`, `selective_power_floor.py` | matching `results/*.json` |
| intervention gate | `gate_role_data_audit.py` | `gate_role_data_audit.json` |
| off-target repair / CFD | `genome_offtarget.py`, `cfd_doench.py`, `cfd_exact_rescore` | `cfd_exact_rescore.json` |
| chromatin recompute + stress | `pdac_residual_foldchange.py`, `h3k27ac_fragility.py`, `h3k27ac_window_and_loci.py`, `h3k27ac_pseudocount.py` | `pdac_residual_foldchange_*.json`, `h3k27ac_*.json` |
| subtype separation | `subtype_resolved_targets.py` | `subtype_resolved_targets.json` |
| tests | `tests/test_cfd_doench.py`, `tests/test_genome_offtarget.py` | 26 assertions, all passing |

*Compiled from the committed record; no number in this document is from memory.*
