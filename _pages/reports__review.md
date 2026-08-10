---
layout: default
title: "Review arc"
description: "The full twenty-eight step review arc that tested and in most cases overturned the claims."
permalink: /reports/review/
group: reports
order: 6
---

An external editorial review of [FINDINGS.md]({{ '/reports/findings/' | relative_url }}) and [METHODS.md]({{ '/reports/methods-full/' | relative_url }}) raised
objections that turned out to be correct. Two of them were tested directly. **One headline claim
does not survive and is retracted below.** This document records what broke, what survives, and
what the corrected scope is.

---

## 1. RETRACTED — "attractor-collapse predicts CRISPR essentiality"

**Previously claimed:** *"Attractor-collapse predicts CRISPR essentiality out-of-modality — AUC
0.653, 50,000-permutation p = 0.0022, beating degree (0.629) and eigenvector (0.584)."*

The reviewer noted that a permutation test against **chance** does not answer whether RAC adds
anything over **degree**, and that the difference (0.024) could be sampling noise. That test had
never been run. It has now been (`scripts/rigorous_validation.py`,
`results/rigorous_validation.json`; 419 genes, 31 positives, positive rate 0.074):

| test | RAC collapse | degree |
|---|---|---|
| AUC | **0.547** | **0.629** |
| PR-AUC | 0.098 | **0.131** (baseline 0.074) |
| **ΔAUC (RAC − degree)** | **−0.082**, 95 % CI **[−0.199, +0.029]**, p = 0.15 | — |
| continuous Spearman vs Chronos | **ρ = 0.051, p = 0.30** | — |
| **partial Spearman given degree + expression + variance** | **ρ = 0.028, p = 0.56** | — |
| 5-fold CV AUC, covariates only → covariates + collapse | **0.653 → 0.652** | — |

**Conclusion: collapse carries no information about CRISPR essentiality beyond degree,
expression level and expression variance.** The partial association is null, and adding collapse
to a covariate model does not improve it (it is marginally worse). The ΔAUC point estimate is
*negative*.

**Why the original number looked better.** Three compounding problems, all of which the reviewer
identified:

1. **The null was against chance, not against degree.** p = 0.0022 only says collapse ranks
   essentials above random — degree does that too, and better.
2. **Model selection reused the endpoint.** The 400-node / τ = 0.4 configuration was chosen by
   its CRISPR AUC and then reported on the same CRISPR labels. "CRISPR held out" was true of
   *parameter fitting* and false of *model selection*.
3. **Configuration sensitivity.** The retest at the selected configuration (motif-free, fixed
   seed) gives 0.547, well below the 0.653 obtained under the ensemble/campaign settings — the
   spread across configurations is comparable to the effect being claimed.

**What is left of the method.** RAC still provides intervention semantics and attractor-state
predictions that a centrality score does not. But the correct statement is now the reviewer's:

> RAC performs **no better than a strong degree baseline** at ranking core-essential genes. Its
> value, if any, is interpretability and control design — not discrimination.

The leave-cell-line-out result (§3) is a *state-reconstruction* result and is unaffected by this,
but it is a diagnostic, not evidence of vulnerability prediction.

---

## 2. ACCEPTED — the chromatin residual is wrongly computed

The residual used ENCODE **signal p-value** tracks. ENCODE distinguishes *signal p-value* from
*fold change over control*; a log-ratio of p-value tracks is **not** a biological fold change.

**Withdrawn:** "+1.596 log2 ≈ 3× more active-enhancer signal."

Also accepted:
- **H3K27ac over TSS ± 2 kb is promoter-proximal signal, not enhancer signal.** Distal enhancers
  must be defined and linked separately before any enhancer claim.
- **PANC-1 is quasimesenchymal** — a basal/mesenchymal model. Validating a *classical* target set
  against it is a substrate mismatch; classical and basal target sets must be analysed separately.
- **Two marks in one cell line are orthogonal assays, not independent biological replication.**
  "Replicated on two independent marks" → "concordant across two orthogonal assays in the same
  cell-line model".
- Culture state, donor, purity, library and batch are confounded with disease status in this
  contrast.

**Required before any chromatin claim:** fold-change-over-control tracks or uniformly processed
counts; expression/accessibility/GC/mappability-matched background; replicate counts and CIs;
subtype-separated analysis; a classical substrate.

---

## 3. ACCEPTED — leave-cell-line-out was not leakage-free, and is demoted

The original LOO reused whole-panel statistics: the co-expression graph was built across all
1,684 lines (including the held-out one), node selection used variance across all 54 PDAC lines,
and per-gene scaling used all 54. The gene-permuted null is also weak — it destroys covariance, so
a model capturing average PDAC architecture beats it trivially.

A leakage-free implementation (node selection, scaling and graph rebuilt from the 53 training
lines inside every fold; nulls = training centroid, rank-5 PCA reconstruction,
covariance-preserving surrogate) is implemented in `scripts/rigorous_validation.py` part C. It had
not completed at time of writing (the machine is heavily contended and the run died on a network
read during part B). **Until it completes, the LOO result is demoted from "strongest evidence" to
"internal diagnostic, leakage not yet excluded".**

`p ≈ 0` is also replaced by an exact statistic or a bound (`p < 1e-15`) in the new implementation.

---

## 4. ACCEPTED — the safety layer failed, and a signed gate now exists

The reviewer's most serious biological objection: **"driver" is not "oncogene to repress"**, and
a safety score that passes TGIF1 repression is not a safety score.

`src/pdac_circuit/attractor/intervention_gate.py` adds a role × direction × subtype gate. Applied
to the top-20 targets (`results/intervention_gate.json`):

| status | genes |
|---|---|
| **allowed** (repression supported) | SETDB1, E2F1, MYBL2, FOSL1 — **4 of 20** |
| **quarantined** | 16 of 20 |
| blocked | 0 in this list (TGIF1 is blocked wherever it appears) |

The quarantine list is itself an indictment of the ranking:

- **BRCA2, ATM, KMT2C** — DNA-repair / chromatin **tumour suppressors**. Repressing them is a
  genome-instability liability, not a therapy.
- **GATA6** — maintains classical epithelial identity; repression risks driving a
  classical → basal switch, i.e. a *more* aggressive state.
- **HNF4G** — stage-dependent; loss can unmask FOXA1-driven metastatic programmes.
- **TGIF1** — blocked outright: its **loss** accelerates KRAS-driven PDAC.
- 10 further genes (ZNF790, SOX13, AHR, AGR2, ZNF331, SF3B1, SMAD3, HOXA3, ZNF528, FAM83A,
  ZNF93, ZNF85) — **unclassified**: no curated role, direction not established.

**The finding this forces:** the convergence score ranks by *attractor movement × disease-up*, so
it systematically surfaces DNA-repair tumour suppressors and uncharacterised zinc fingers. **The
target list, as ranked, is not a list of therapeutic repression candidates.** Only 4/20 have a
defensible repression direction. `safety` is renamed **heuristic feasibility/risk score**; it is
not calibrated against adverse-direction interventions.

---

## 4b. DONE — targets re-ranked *through* the gate, not after it

The gate was originally a post-hoc filter on a score that ranked by attractor-movement x
disease-up. That ordering is what produced a list led by DNA-repair tumour suppressors. The
pipeline is now inverted (`scripts/gated_target_ranking.py`, `results/gated_target_ranking.{json,md}`):

1. the admissible direction is decided **first**, from gene role;
2. evidence is **signed for that direction** — a tumour suppressor scores for CRISPR**a**, and
   being disease-*down*, deleted or hypermethylated counts in its favour, the exact opposite
   evidence to an oncogene;
3. **collapse carries zero weight**, because its discrimination did not beat degree (§1). The
   ranking is built only from independently measured evidence: direction-appropriate expression
   change, copy number, promoter methylation, protein support, chromatin concordance.

Of 20 input targets: **4 rankable, 4 quarantined, 12 unclassified.**

| gene | modality | direction-aware score | role |
|---|---|---|---|
| SETDB1 | CRISPRi | **+0.778** | chromatin oncogenic |
| MYBL2 | CRISPRi | **+0.714** | proliferation oncogenic |
| E2F1 | CRISPRi | **+0.692** | proliferation oncogenic |
| FOSL1 | CRISPRi | +0.418 | AP-1 oncogenic |

All four survivors are oncogenic/proliferation factors — a coherent CRISPRi direction. **This is
the usable output of the project: four hypotheses, not twenty.** The remaining 16 are either
state/stage-dependent (GATA6, HNF4G, BRCA2, ATM, KMT2C) or have no established direction.

That 16/20 of a "convergent target catalogue" cannot be acted on is the honest measure of how much
the original ranking was driven by attractor movement rather than by therapeutic logic.

---

## 5. ACCEPTED — claim language

| was | now |
|---|---|
| "cell death is literally collapse to the dead attractor" | attractor collapse is a **dynamical vulnerability proxy** (the dead state is partly imposed by the loss) |
| "targets sit on PDAC-gained chromatin" | targets show **PANC-1-versus-healthy chromatin concordance** |
| "replicated on two independent marks" | concordant across two **orthogonal assays in the same cell-line model** |
| "3× more active-enhancer signal" | **withdrawn** |
| "protein-level TF activity" | **protein abundance and detection support** |
| "real circuit" / "cert: real" | **real-data-calibrated in-silico circuit candidate** |
| "safety" | **heuristic feasibility/risk score** |
| "off-risk 0.00" | **no candidate off-target detected under the bounded search** |
| "the first PDAC disease residual" | the first **disease-contrast analysis within this pipeline** |
| "regulatory coupling" | **expression-derived phenomenological coupling** |
| "CRISPR held out" | **CRISPR excluded from parameter fitting but used for model selection and evaluation** |
| "no synthetic data" | no synthetic **training observations** (generated promoters and designed circuits are synthetic by construction) |

Also accepted: the 6,006 simulated circuits are **not** 6,006 experiments; robustness = 1.00
everywhere, safety in a 0.86–0.90 band and off-risk uniformly 0.00 indicate **metric saturation**,
not circuit quality. The graph is **not** strictly a TF network — several nodes are not TFs.
Seed `20260808` is **planned**, not completed.

---

## 6. What actually survives

| result | status |
|---|---|
| Bistable attractor formulation with intervention semantics | **stands** as a method contribution; not the first attractor-control approach (CellOracle, CellBox, RACIPE precede it) — novelty is the specific combination |
| Collapse predicts CRISPR essentiality | **RETRACTED** — no gain over degree; partial association null |
| Leave-cell-line-out state reconstruction | **demoted** to internal diagnostic; leakage-free rerun pending |
| PANC-1 chromatin concordance | **stands directionally**, but must be recomputed on fold-change tracks, subtype-separated, background-matched |
| Long-range model r ≈ 0.71 (8 seeds) | **stands as optimisation stability only** — needs leakage-proof chromosome split and real baselines; belongs in a separate paper |
| PDAC-selective dependency scarcity (7/1,164) | **stands** as a benchmark-design lesson, not as "selective dependencies do not exist" |
| Two pipeline defects (silent no-op, chromosome-order) | **stand** — software findings |
| KLF5 / GATA6 recovery | **positive-control recovery**, not discovery |
| Signed intervention gate | **new** — and it invalidates most of the ranked list |

**The defensible nucleus, in the reviewer's words:**

> An expression-calibrated bistable attractor model provides an interpretable substrate for
> intervention design. Its vulnerability score does **not** outperform network degree.

---

## 7. What's next, in priority order

1. **Finish the leakage-free LOO** (part C) and report it with exact statistics against the
   centroid / PCA / covariance-preserving nulls. If it survives, it is the method's best evidence.
2. **Recompute the chromatin contrast** on fold-change-over-control tracks, background-matched,
   subtype-separated, with replicate counts and CIs. Add a classical substrate (organoid or a
   classical line) — PANC-1 alone cannot validate a classical target set.
3. **Re-rank targets through the signed gate**, not after it. Add CRISPRa for tumour suppressors
   and state-maintaining factors. Report `allowed` candidates only.
4. **Selection-aware null + nested CV** for any retained discrimination claim, and predeclare the
   primary endpoint and essentiality threshold.
5. **Fix Module I the same way** — MCDA weights are grid-searched to maximise recovery of the same
   seven drivers that are then reported; the p = 0.003 is anti-conservative unless the whole
   search is repeated inside every permutation.
6. **Break the metric saturation** — global parameter uncertainty, Sobol/Latin-hypercube
   sensitivity, adversarial regimes, negative-control circuits, memorisation tests for generated
   promoters.
7. **Narrow the paper** to the RAC method; move Modules I–VII, the GAN, PDACircuitFormer, Hi-C,
   single-cell and the defects to supplement or separate work.
8. **Keep the seal.** Freeze the model, hyperparameters and evaluation plan with an immutable
   record, then run the protected studies **once** and publish every outcome.
9. **One functional validation tier** — an anchor (KLF5) plus a discovery-like candidate
   (TCF7L2 / YBX3) in subtype-appropriate models, multiple reagents, nonmalignant controls.

**Suggested title, once corrected:** *A bistable attractor-control model prioritises
transcriptional vulnerabilities in pancreatic ductal adenocarcinoma* — without "synthetic gene
circuits" until a circuit is actually tested.

---

## 8. Standing assessment

The review's editorial verdict — **major revision, potentially publishable** as a computational
systems-biology methods paper; **not** as a target-discovery or circuit-efficacy paper — is
accepted in full. The retraction in §1 lowers it further: the primary quantitative claim is gone,
and what remains is a method with intervention semantics whose discrimination is not better than
degree.

That is a worse result than was reported yesterday, and it is the correct one.

---

## 9. Subtype separation (review item 3): performed, and the demanded lists do not exist

The review required that "the classical and basal target lists be analyzed separately," and stated
that PANC-1 is quasimesenchymal and therefore a poor substrate for a classical target set. Both
points are now tested against data rather than argued.
Script: `scripts/subtype_resolved_targets.py` · result: `results/subtype_resolved_targets.json`.

**A defect found in our own first attempt.** Assigning each line by `argmax(basal, classical)` — the
obvious rule, and the one first written here — forces every cell line into a subtype, including
lines that express *neither* programme above the panel mean. Those are precisely the
quasimesenchymal lines. The rule was replaced: a line is called only if it is above the panel mean
on the winning programme **and** the two scores are separated by > 0.20; otherwise it is held out.
Of 54 scored DepMap PDAC lines this yields **22 basal / 13 classical**, with **14 intermediate and
5 ambiguous held out** rather than silently assigned.

**PANC-1 is intermediate, and the reviewer is right for a reason the reviewer did not give.**
PANC-1 scores **basal −0.344 / classical −0.169** — *below* the panel mean on **both** programmes.
Argmax would have labelled it "classical," which is how a quasimesenchymal line gets mistaken for a
classical model. It is neither. Every PANC-1 chromatin concordance number in this repository is
therefore a measurement in a line that faithfully represents **neither** target set, which is a
stronger substrate objection than the original review made, and it is conceded.

**No subtype-specific candidate list exists.** Across the 8 gated genes, differential Chronos
(basal vs classical, Mann-Whitney, Benjamini-Hochberg over 8 tests):

| gene | status | Chronos basal | Chronos classical | diff | p | q (BH) |
|---|---|---|---|---|---|---|
| GATA6 | quarantined | +0.012 | −0.196 | +0.208 | 0.019 | 0.154 |
| BRCA2 | quarantined | −0.444 | −0.347 | −0.097 | 0.097 | 0.387 |
| FOSL1 | **candidate** | −0.639 | −0.481 | −0.159 | 0.327 | 0.871 |
| SETDB1 | **candidate** | −0.523 | −0.469 | −0.054 | 0.674 | 0.984 |
| E2F1 | **candidate** | −0.492 | −0.492 | +0.000 | 0.857 | 0.984 |
| MYBL2 | **candidate** | −0.724 | −0.683 | −0.041 | 0.984 | 0.984 |

**Not one gene survives multiple-testing correction.** The four candidates are **pan-PDAC**: they
are essential in both subtypes, at effectively the same magnitude. There is no basal list and no
classical list to separate, so the review's request is answered by showing the premise does not
hold for these targets — not by producing two lists.

**This is not a threshold artefact.** Re-running across margins 0.0–0.4 (line splits from 26b/14c
to 21b/11c) leaves every candidate non-significant at every margin (SETDB1 0.67–0.93, MYBL2
0.74–1.00, E2F1 0.62–0.93, FOSL1 0.33–0.67). The null is stable to the one free parameter.

**The split itself works — the positive control fires.** GATA6, the canonical classical-identity
factor, is the single most subtype-differential gene tested and is more essential in classical
lines (p = 0.019), recovered blind from expression alone. It does not clear BH at q = 0.154, so it
is reported as a **directionally correct positive control, not a finding**. That it ranks first,
with the correct sign, is evidence the assignment is measuring real subtype biology; that it fails
correction is evidence the panel is underpowered for this contrast (13 classical lines).

**Consequence for the constructs.** The four guides are correctly described as **pan-PDAC
candidates**. They should *not* be advertised as subtype-targeted, and — following the PANC-1
finding above — should be validated in lines that actually sit at the poles of the signature, not
in PANC-1, whose intermediate state makes it uninformative about either target set.

---

## 10. Module I selection reuse (review item 5): the null was optimistic, the claim survives anyway

The review flagged that the MCDA weights are grid-searched to maximise recovery of the same seven
drivers that are then reported, so `p = 0.003` is anti-conservative "unless the whole search is
repeated inside every permutation." It is the identical defect that sank the attractor-collapse
claim: `prioritize_targets` picks the weights using the true driver labels (`prioritize.py:95`),
then permutes those labels while holding the weights **fixed** (`prioritize.py:114`). The weights
already encode where the drivers are, so no permuted replicate can ever compete on equal terms.

`scripts/module1_selection_aware_null.py` re-runs the **entire** grid search inside each of 2,000
permutations — same recovery@k objective, same most-balanced tie-break — so every replicate gets
the weights that are best for *its own* shuffled labels. The naive fixed-weight null is computed
on identical draws, so the inflation is measured rather than argued.

| statistic | observed | naive p (weights frozen) | **selection-aware p** | naive null mean | selection-aware null mean |
|---|---|---|---|---|---|
| recovery@k (k=10) | 2 / 7 | 0.0015 | **0.0130** | 0.053 | **0.326** |
| mean rank of controls | −298.7 | 0.0045 | **0.0075** | −662.4 | −653.7 |

**The selection advantage is real and large.** Under the naive null a random label set recovers
0.05 drivers on average; once each replicate may choose its own weights, that rises to **0.326** —
a null six times more competitive. The recovery@k p-value inflates **8.7×** (0.0015 → 0.0130).

**But the finding holds.** Both statistics remain significant at α = 0.05 under the honest null
(0.0130 and 0.0075). Unlike the collapse claim, this one does not depend on the flaw. The reported
p-value was wrong; the conclusion was not. It is corrected here rather than withdrawn.

**Two things that should temper it.** The effect size is thin: **2 of 7** drivers in the top 10,
against a prereg margin of exactly 2 — it clears by nothing. And the selected weights are
`c_subtype 0.50, c_expr 0.25, c_breadth 0.25, c_spec 0.00, c_onc 0.00`: the grid search assigns
**zero weight to both the oncogenicity and the tumour-specificity criteria**, so the
"multi-criteria" ranking is in fact driven by three of its five criteria, and driver recovery is
carried mostly by subtype correlation. That is worth stating plainly before anyone reads the
p-value as an endorsement of the MCDA design.

---

## 11. Metric saturation (review item 6): the cause is not what we assumed

The review required that the saturated objectives be broken open (robustness = 1.00 everywhere,
off-target risk = 0.00 everywhere). `scripts/robustness_metric_diagnostic.py` tests two hypotheses
against negative controls and the repository's own golden fixtures.
Result: `results/robustness_metric_diagnostic.json`.

**Hypothesis 1 — refuted, and it was ours.** We assumed `parameter_sweep`'s single global
multiplicative factor per axis (applied to *every* gene at once) made it near-incapable of breaking
a circuit. It is not: the sweep scores `robust_circuit` at **1.000** and both `fragile_circuit` and
`repressilator` at **0.203**, a spread of **0.80**. It detects instability perfectly well, and it
correctly fails an oscillator. The global-vs-per-gene distinction is not the defect.

**Hypothesis 2 — confirmed, and it is the real defect.** `steady_state_within_tol` asks only
whether the system *settles*, never whether it settles to the value the circuit was designed to
compute. Setting every gene's output drive to zero — a circuit that provably cannot compute
anything — yields robustness **1.000**, the maximum, for **every** fixture tested. The metric's
optimum is achieved by doing nothing. `robustness = 1.0` is therefore not evidence that a circuit
works, and since every circuit the pipeline actually delivers (`run_basal`, `run_classical`,
`gated_constructs`) scores exactly **1.0**, the objective contributes **no ranking information** to
the NSGA-II front at all. Module VI is optimising over three live objectives, not four.

**The repair, demonstrated.** Scoring instead against agreement with the circuit's own compiled
Boolean fixed points — the spec `boolean.py` already derives — restores discrimination exactly
where the shipped metric saturates:

| circuit | shipped (settle-only) | Boolean-correctness |
|---|---|---|
| `robust_circuit` | 1.000 | **1.0** |
| `toggle_switch` | 1.000 | **1.0** |
| `monostable_circuit` | 1.000 | **0.0** |
| `fragile_circuit` | 0.203 | 0.0 |
| `repressilator` | 0.203 | 0.0 |

The shipped metric cannot separate `robust`, `monostable` and `toggle` — all three are 1.000. The
correctness criterion does, and it still fails the oscillator and the fragile design, which is the
required behaviour.

**Two honest caveats.** `monostable_circuit` settles on 100% of draws yet matches its Boolean fixed
point on none; that is either a genuine mismatch between its ODE and its own spec or a residual
thresholding artefact in the diagnostic, and it is **unresolved** — it is not being reported as a
finding about that circuit. Separately, a first version of this diagnostic required a *unique*
Boolean fixed point, which silently resolved the reference to `None` for every multistable circuit
and scored them all 0.0; the toggle switch is bistable by construction. That bug produced a clean
"every circuit is wrong" table that was entirely an artefact, and it is recorded here because it is
exactly the failure mode this section exists to catch.

**Not yet done:** the same treatment for `off_target_risk = 0.00`, and Sobol/global-sensitivity
attribution over the corrected objective.

---

## 12. GAN memorisation (review item 6): no memorisation, but the strength scale is not calibrated

The review required a memorisation test for the generated promoters. The shipped certification
cannot serve as one: it passes the GAN because its 4-mer spectrum is closer to real promoters
(JS 0.0088) than random DNA is (JS 0.0508) — but **memorisation minimises JS**. A generator that
simply copied its training set would score JS ≈ 0 and pass most convincingly of all. "Realism
beats random" is evidence the model learned something, not evidence it invented anything.
Script: `scripts/gan_memorisation_test.py` · result: `results/gan_memorisation_test.json`.

**No memorisation — the GAN passes cleanly.** Nearest-neighbour identity on aligned 1,024 bp
TSS-anchored windows, 500 generated vs 4,000 training promoters:

| set | median NN identity to training set | p95 | max |
|---|---|---|---|
| generated | 0.3154 | 0.3291 | 0.3584 |
| real promoters (leave-one-out) | 0.3262 | 0.3653 | 0.6221 |
| random DNA | 0.2998 | — | — |

Generated sequences are **not** closer to the training set than a real promoter is to its nearest
training neighbour — they are marginally *further* (Mann-Whitney p = 1.000 in the tested
direction), with **zero** near-exact copies and a maximum identity of 0.358 against 0.622 for real
promoters. The novelty claim survives, on a test that could have refuted it.

**A fairer reading of the strength claim than the one shipped.** The reported `strength_uplift =
0.029` is measured against random DNA, which is not the baseline a part-design tool must beat.
Scoring generated, real and random sequence side-by-side through the same Module II model:

| set | median strength | p90 |
|---|---|---|
| real promoters | 0.8000 | 0.9692 |
| generated | 0.7895 | 0.9136 |
| random DNA | 0.7583 | 0.8910 |

The generator lands **0.011 below real promoters** and **0.031 above random**, i.e. it closes
**~75%** of the random→real gap. That is a considerably better result than "uplift 0.029" conveys,
and it is the number that should be reported.

**But it exposes a third saturated metric.** The entire dynamic range between random DNA and real
promoters is **0.042** on a nominally [0,1] scale: the Module II model scores *random DNA* at a
median of **0.758**. The unit strength is therefore not calibrated, and an absolute figure like
"predicted strength 0.79" carries almost no information — chance sequence already scores 0.76.
Every downstream use of promoter strength as an absolute quantity (including `pred_strength_gen_max
= 0.993`) inherits this. Percent-of-gap-closed is meaningful here; the raw score is not.

This joins robustness (§11) and off-target risk on the list of objectives that need recalibration
before Module VI's Pareto front can be said to be ranking on four real axes.

---

## 13. off_target_risk = 0.00 is blind, not clean — and the four constructs must be re-described

§11 left `off_target_risk = 0.00` untested. It is now tested, and it does not survive.

**The search space is the defect.** `design.py:71-75` builds the off-target search set as the
target loci **±5 kb**, plus any caller-supplied background:

```python
search_seqs = background_search or []
for locus in loci:
    s = fetch_sequence(locus["chrom"], max(0, locus["start"] - 5000), locus["end"] + 5000)
    search_seqs.append((locus["chrom"], s))
```

For a four-target design that is ~40 kb against a 3.1 Gb genome — **0.0013%**. The pipeline looks
for off-targets only in the immediate neighbourhood of the gene it is already cutting. Finding
none there is close to guaranteed by construction. `offtarget.py` is candid that exhaustive search
is out of scope; the danger is that the resulting `0.00` reads as safety.

**A real scan of the local hg38.** `scripts/genomewide_offtarget_audit.py` scans all 3.09 Gb of the
main assembly, both strands, for every NGG site within ≤4 mismatches of each proposed guide, and
scores hits with the repository's **own** MIT/CFD-style functions — same scorer, honest search
space. Result: `results/genomewide_offtarget_audit.json` (138 s).

| gene | shipped report | perfect matches (incl. on-target) | **off-targets ≤4 mm** | 2 mm | 3 mm | 4 mm |
|---|---|---|---|---|---|---|
| SETDB1 | 0 off-targets, CFD 1.00 | 1 | **206** | **1** | 17 | 188 |
| MYBL2 | 0 off-targets, CFD 1.00 | 1 | **75** | 0 | 3 | 72 |
| E2F1 | 0 off-targets, CFD 1.00 | 1 | **68** | 0 | 6 | 62 |
| FOSL1 | 0 off-targets, CFD 1.00 | 1 | **35** | 0 | 3 | 32 |

**384 off-target sites across the four guides, every one of them reported as zero.** SETDB1 carries
a **2-mismatch** site. FOSL1 has a sampled hit at chr1:7,174,801 (`TCTGCCTCGCCTGGGCCGTGGGG`) scoring
**CFD-style 0.921** — predicted ~92% as active as the intended target.

**One real positive.** Every guide has **exactly one** perfect genomic match: each is uniquely
placed at its intended locus, with no second perfect site anywhere in the assembly. That is a
genuine specificity result, and it is the only part of the original specificity claim that holds.

**Limitation of this audit, stated rather than buried.** The CFD figures above are computed over a
capped sample of ≤12 hits per guide, not all 384; the true worst-case CFD is therefore likely
*worse* than shown, not better — a 12-hit sample already surfaced 0.921. Bulges are not modelled
(substitutions only), and the CFD implementation remains position-granular rather than the exact
Doench-2016 nucleotide-pair matrix.

**Consequence.** The four constructs in §9 were previously described as having "no candidate
off-target detected under the bounded search". That phrasing was accurate about the procedure and
misleading about the biology: the bounded search could not have detected one. They must be
described as **uniquely placed at their intended locus, with 35-206 genome-wide near-matches at
≤4 mismatches each, unranked and unvalidated**. No guide here is ready to order on specificity
grounds, and FOSL1's high-CFD site would need direct assessment first.

**The fix to the code** is to feed the real assembly into the search rather than a ±5 kb window,
and to report the CFD aggregate over genome-wide hits. Until then `cfd_specificity` and
`off_target_risk` should not be emitted at all, rather than emitted as 1.00 and 0.00.

---

## 14. The off-target defect is repaired in the code, and it rejects our own candidates

§13 documented the flaw. Documenting a defect while the tool keeps emitting `0.00` is half a job,
so the search is now fixed rather than merely described.

**What changed.** New module `src/pdac_circuit/grna/genome_offtarget.py` scans the whole local hg38
(both strands, ≤ `max_mm` mismatches) using the project's own MIT/CFD scorers. `design_guides` now
treats the ±5 kb neighbourhood search as what it is — a cheap **pre-filter for ordering
candidates** — then rescores the shortlist genome-wide and **applies the pre-registered gate to
those real numbers**. The marginal cost is small: encoding the assembly dominates, so ~12 guides
costs little more than 4 (~140 s).

The failure path matters as much as the happy path. Previously, an absent assembly still produced
a bounded `cfd_specificity = 1.00`. Now specificity is emitted as `None`, and `off_risk` (already
`1.0 - (spec or 0.0)`) fails **safe** to 1.0. Absent evidence reads as maximum risk, not as
perfect safety. The envelope also carries `offtarget_scope`, so a reader can see whether a number
came from the genome or from a 40 kb window.

**Two independent implementations agree exactly.** The standalone audit (§13) and the library
module return identical counts — 75 / 206 / 35 / 68 — on separate code paths.

**The repaired gate rejects all four of our own candidates.**

| gene | off_risk (shipped) | **off_risk (genome-wide)** | cfd_specificity | prereg gate ≥ 0.5 |
|---|---|---|---|---|
| SETDB1 | 0.00 | **0.958** | 0.042 | **FAIL** |
| MYBL2 | 0.00 | **0.889** | 0.111 | **FAIL** |
| E2F1 | 0.00 | **0.819** | 0.181 | **FAIL** |
| FOSL1 | 0.00 | **0.782** | 0.218 | **FAIL** |

`design_guides` now abstains (`certified-negative`) where it previously returned a confident guide
with `off_risk = 0.00`. Every construct advertised earlier in this repository is withdrawn on
specificity grounds.

**A caveat that cuts against these new numbers, not for them.** `cfd_style_score` uses Hsu
*position* weights as a proxy for the Doench-2016 *nucleotide-pair* matrix. Position-only weighting
penalises every mismatch at a position equally, whereas the true CFD matrix scores many specific
mismatch identities near zero. Summing that proxy over 206 sites therefore likely **over**-states
total off-target activity, so `cfd_specificity = 0.042` is probably **pessimistic** and these
guides may not be as bad as the table implies. The honest position is that the true value lies
between the old `0.00` and these figures, and pinning it down requires the exact Doench-2016
matrix — which remains unimplemented.

What improved is not that we now have the right number. It is that **the direction of the error
changed from fails-dangerous to fails-safe**, and that the reported quantity is now a property of
the genome rather than of the search radius.

### 14b. Correction to §14: the rejection is not robust either

> **SUPERSEDED BY §24 — and its central prediction was wrong.** The `undetermined` verdict below
> is now RESOLVED with the exact Doench-2016 matrix (downloaded, validated against 7 published
> test vectors). Two claims below do not survive: (i) *"at ≤3 mm three of four guides pass"* —
> in fact only FOSL1 passes; MYBL2 (0.659→0.469) and E2F1 (0.510→0.447) flip to FAIL; and
> (ii) *"the position-granular figures are likely pessimistic"* — the approximation erred in
> BOTH directions and was **optimistic** at the decision-relevant ≤3 mm cutoff. See §24.

§14 concluded that the repaired gate "rejects all four candidates" and that every construct is
"withdrawn on specificity grounds". That is an over-correction, and the sensitivity analysis it
asked for refutes it. `scripts/offtarget_cutoff_sensitivity.py` recomputes aggregate specificity
restricted to ≤2, ≤3 and ≤4 mismatches — separating conclusions that survive the position-granular
CFD approximation from those carried by the distant tail it models worst.

| gene | ≤2 mm (n, spec) | ≤3 mm (n, spec) | ≤4 mm (n, spec) | gate ≥ 0.5 |
|---|---|---|---|---|
| SETDB1 | 1, **0.862** PASS | 18, **0.283** FAIL | 206, 0.042 FAIL | fails from ≤3 mm |
| FOSL1 | 0, 1.000 PASS | 3, **0.707** PASS | 35, 0.218 FAIL | fails only at ≤4 mm |
| MYBL2 | 0, 1.000 PASS | 3, **0.659** PASS | 75, 0.111 FAIL | fails only at ≤4 mm |
| E2F1 | 0, 1.000 PASS | 6, **0.510** PASS | 68, 0.181 FAIL | fails only at ≤4 mm |

**No guide is rejected at every cutoff.** For MYBL2, E2F1 and FOSL1 the rejection rests *entirely*
on the 4-mismatch tail: 72 of MYBL2's 75 sites, 62 of E2F1's 68, and 32 of FOSL1's 35 sit at 4
mismatches, where the true CFD depends on *which* substitutions occurred — exactly the information
Hsu position weights discard. At ≤3 mm all three clear the gate.

**The honest verdict is therefore `undetermined`, not `rejected`.** Both of this repository's
previous positions were overconfident in opposite directions:

* `off_risk = 0.00` — wrong, and dangerously so: the search covered 0.0013% of the genome.
* `off_risk = 0.78–0.96, all rejected` — also unsupported: driven by a tail the scorer approximates.

**What survives regardless of the approximation:**
* every guide is **uniquely placed** — exactly one perfect genomic match each;
* **SETDB1 is the one robust concern** — it fails at ≤3 mm (spec 0.283, 18 sites) and carries a
  **2-mismatch** site, which no CFD refinement will explain away;
* **E2F1 at ≤3 mm is marginal** (0.510 against a 0.5 gate) — it clears by 0.010 and should not be
  called specific;
* MYBL2 (0.659) and FOSL1 (0.707) are the most defensible of the four at ≤3 mm — though FOSL1 still
  owns the single highest-CFD site found (0.921).

**This does not resolve the true value.** Only the exact Doench-2016 nucleotide-pair matrix can, and
its coefficients are deliberately **not** reconstructed from memory here — inventing them would be
precisely the fabrication this project forbids, and CRISPOR's distribution is a pickle (loading it
is arbitrary code execution, so it is not used). Until that matrix is obtained from a safe source,
`off_target_risk` should be reported **with its mismatch cutoff attached**, and no guide in this
repository should be described as specific or as rejected.

### 14c. Cost of the repair, and an unfixed architectural problem

Correctness first: the restructured scanner returns counts **identical** to the independent audit
(75 / 206 / 35 / 68) and passes 14 planted-site reference tests, including the minus-strand
regression. Nothing below trades accuracy for speed.

A genome-wide scan is inherently expensive (~2 min: read + encode 3.1 Gb, PAM-mask ~3.09e9 windows
on both strands, then 20 mismatch passes per guide). Two optimisation hypotheses were tried and
**both were wrong**, which is worth recording because each was plausible:

1. *"Re-encoding the assembly dominates."* Caching the encoded chromosomes
   (`_encoded_chrom`, ~3.1 GB resident, `PDAC_NO_GENOME_CACHE=1` to disable) recovered only
   **1.5x** (59.6 s → 38.9 s). Encoding was not the cost.
2. *"The per-chunk gather dominates."* Hoisting `w[sub, lo:lo+20]` out of the per-guide loop so all
   guides share one gathered block gave **7%** (138 s → 129 s). Also not the cost.

The residual is the PAM masking itself, which is per-block fixed work that no guide-loop
restructuring can amortise. Both changes are kept (cleaner, slightly faster, identical results),
but they did **not** solve the real problem.

**The unfixed problem.** `run_pipeline` designs guides one target at a time
(`orchestrator` → `select_repressor` → `design_guides`), so a k-target run pays k separate
genome scans: `run_pipeline(top_k=6)` now costs ~11 min where it previously cost seconds. This is a
direct regression introduced by the repair, and it is **real**, not incidental — `tests/
test_pipeline.py::test_pipeline_end_to_end` is the visible casualty.

The correct fix is architectural: the scan is a property of the *genome*, not of a guide, so the
orchestrator should collect every target's shortlist and issue **one** scan for all of them,
rather than one per target. That is not done here. Until it is, `run_pipeline` is slow by design,
and the alternative — defaulting `genome_wide_offtarget=False` to make it fast again — is exactly
the trade this repository must not make: it would return `certified-negative` guides quickly
instead of honest ones slowly.

A note on how this was found: the slowdown was first reported here as "the suite stalls on
data-heavy tests unrelated to my change." That was asserted without checking and was false. The
targeted run that looked green (`-k "grna or design or offtarget or repressor or parts"`) does not
match test *names* containing "pipeline", so it silently skipped the one caller the change most
affects — a filter that excludes the important case is indistinguishable from one that covers it.

### 14d. Correction to §14c: batching is NOT the fix either (measured)

§14c asserted "the correct fix is architectural: the orchestrator should issue one scan for all
targets rather than one per target." That was a third un-measured performance claim, and measuring
it refutes it. Scan cost vs guide count, with the genome cache already warm (so this isolates scan
scaling from disk):

| guides | total | per guide |
|---|---|---|
| 1 | 45.9 s | 45.95 s |
| 2 | 62.5 s | 31.26 s |
| 4 | 103.8 s | 25.95 s |
| 8 | 178.2 s | 22.28 s |
| 16 | 360.8 s | 22.55 s |

**Cost ≈ 25 s + 21 s × n — linear in the number of guides.** The marginal per-guide term dominates
the fixed term for any realistic shortlist. Batching a 6-target run (72 guides) into one scan costs
`25 + 72×21 = 1537 s` against `6 × (25 + 12×21) = 1662 s` for separate scans: a **7.5% saving**.
Batching is nearly useless, and §14c's architectural recommendation is withdrawn.

(These absolute figures were collected while the test suite was also running, so they are upper
bounds inflated by contention — a methodological error, since both jobs scan the genome and each
caches ~3.1 GB. The linear *shape*, which is what the conclusion rests on, is unaffected: both
terms inflate together.)

**Where the time actually goes:** 20 mismatch passes over ~380 M PAM sites, per guide. The fix must
reduce per-guide work, not reorganise call sites. The obvious candidate is early termination: a
site needs ≤4 mismatches, but after only 8 positions ~89% of sites already exceed that budget
(X ~ Binomial(8, 0.75), P(X≤4) ≈ 0.11), so the remaining 12 passes are ~89% wasted. Pruning
survivors periodically should recover roughly 2x. **That is a hypothesis and is not implemented
here — it is recorded as a hypothesis precisely because the previous three were stated as
conclusions and all three were wrong.**

**Tally of performance claims in this section, for calibration:**

| claim | predicted | measured |
|---|---|---|
| re-encoding the assembly dominates | large win | **1.5x** |
| the per-chunk gather dominates | large win | **7%** |
| batching scans across targets is the fix | large win | **7.5%** |
| early termination in the mismatch loop | ~2x | **untested** |

Three plausible, well-reasoned, wrong. The lesson is not about this scanner: it is that
*"I reasoned about where the cost is"* is worth nothing next to a measurement — the same
relationship that holds between a bounded search and a genome-wide one (§13), between
"does it settle" and "does it compute the right answer" (§11), and between JS divergence and a
memorisation test (§12).

### 14e. Fourth performance hypothesis: exact, and slower. Reverted.

§14d predicted that early termination in the mismatch loop would recover ~2x, and flagged it as a
hypothesis rather than a conclusion. It was implemented and measured. It is **wrong too**, and in
the worst direction: **9% slower** (140.4 s vs 129.0 s on the same four guides), with counts
identical (75 / 206 / 35 / 68) and all 14 reference tests passing.

The reasoning was sound and the implementation was *exact* -- mismatches only accumulate, so a site
over budget at position 5 can never return under it, and ~89% of sites do exceed a 4-mismatch
budget within 8 positions. The flaw is that `mm[alive] += (km[alive, p] != gd[p])` performs a
gather and a scatter per position, and NumPy fancy indexing costs more than the contiguous
comparison it removes. The saved work was already cheap; the bookkeeping added to avoid it is not.

**Reverted.** A comment now marks the straight loop as deliberate, so the next reader does not
re-derive the same attractive, wrong idea.

**Final tally — every performance claim made about this scanner:**

| claim | predicted | measured | outcome |
|---|---|---|---|
| re-encoding the assembly dominates | large win | 1.5x | kept (marginal) |
| the per-chunk gather dominates | large win | 7% | kept (marginal) |
| batching scans across targets is the fix | large win | 7.5% | withdrawn (§14d) |
| early termination in the mismatch loop | ~2x | **-9%** | **reverted** |

**Four for four wrong.** Each was plausible, each was reasoned from the code, and not one survived a
stopwatch. The genome-wide scan costs ~25 s + ~21 s per guide because that is what comparing a
20-mer against ~380 M PAM sites costs; there was no waste to reclaim, only an intuition that there
must be.

This is the same lesson as §11, §12 and §13, applied to the author rather than the codebase: a
confident account of a mechanism is not evidence about that mechanism. The off-target search
reported 0.00 because nobody measured what it searched. The robustness metric reported 1.00 because
nobody asked a dead circuit. The GAN passed because nobody tested for copying. Here, four
optimisations were "obviously" needed because nobody timed the thing first.

**Consequence for the regression (§14c).** It is real and worse than first stated: `test_pipeline.py`
does not complete a single test in 40 minutes (`EXIT=124`, uncontended). The arithmetic matches --
`select_repressor` passes `top_k=3`, so the shortlist is `max(3*3, 12) = 12` guides at
`25 + 12x21 = 277 s` per target, times 6 targets = ~28 min for `test_pipeline_end_to_end` alone.
No optimisation above closes that gap, because the cost is intrinsic.

The honest options are therefore all trade-offs, not fixes:
* scan fewer candidates per target (shortlist `top_k` rather than `max(top_k*3, 12)`) -- 4x faster,
  at the cost of never re-ranking a candidate the bounded pre-filter mis-ordered;
* mark the pipeline tests slow and accept ~30 min;
* keep `genome_wide_offtarget=False` for tests and assert `certified-negative` -- honest, since a
  test that is not testing specificity should not claim it.

None is implemented here. Choosing among them is a judgement about what the pipeline is *for*, and
that belongs to the maintainer, not to a performance argument.

### 14f. What the suite actually says (measured, uncontended)

Every claim in §14c about *which* tests were slow and *why* was made without running them. Run
properly (`pytest tests/ --ignore=tests/test_pipeline.py -q --durations=10`, nothing else running):

**183 passed, 1 failed, of 184.** The change breaks nothing.

| observation | attribution |
|---|---|
| 183/184 pass with the off-target repair in place | the repair is sound |
| `test_signal.py::test_real_bam_coverage_enriched_at_active_gene` FAILS: `ModuleNotFoundError: No module named 'bamnostic'` | **pre-existing, not this work** — `signal/bamio.py` imports `bamnostic`, which is not declared in `pyproject.toml`, and the diff touches no file under `src/pdac_circuit/signal/` |
| `test_attractor.py::test_full_attractor_control_real_data` = **846 s (14 min)** | **pre-existing** — this is the "stalls at 4 dots" the earlier runs hit |
| `test_pipeline.py` (2 tests) does not finish in 40 min | **this work** — verified in isolation (`EXIT=124`) |

**Both earlier accounts were half-right, which is why neither should have been stated.** The first
("the suite stalls on data-heavy tests unrelated to my change") was correct about `test_attractor`
and wrong to generalise. The second ("no -- it is my change") was correct about `test_pipeline` and
wrong about the stall it was actually explaining. There are two slow tests with two distinct
causes, and each account attributed both to whichever it had just noticed. Reasoning produced a
coherent story either way; only `--durations` distinguished them.

**Incidental defect found, not fixed:** `bamnostic` is imported by `signal/bamio.py` but absent
from the declared dependencies, so a clean install cannot run `test_signal`. Out of scope here and
recorded rather than silently patched.

**Net state of the repair:** the off-target fix is verified correct (two independent
implementations agreeing on 75/206/35/68, 14 planted-site reference tests, 183/184 suite) and
carries one real, intrinsic cost: `run_pipeline` designs guides per target, and a genome-wide scan
is ~25 s + ~21 s/guide, so a 6-target run is ~28 min. That trade -- honest slow answers over fast
false ones -- is the maintainer's to make, and the options are in §14e.

---

## 15. Where the collapse signal lives: a conditional deepening of the §1 retraction

§1 retracted the essentiality claim on a *global* statistic (partial Spearman 0.028, p 0.56). A
global average is the wrong lens for "does the model retain any value", because in a co-expression
graph degree ≈ essentiality almost tautologically (hubs are housekeeping), so the pooled test is
dominated by genes where degree is trivially informative and collapse cannot add anything. The
sharper question is conditional: **among genes of comparable degree, where degree itself cannot
separate essential from non-essential, does collapse rank the essential ones higher?**

`scripts/conditional_collapse_signal.py` answers it with two views on the *same* primary
configuration as part A (400 nodes, tau 0.4 — so this extends the retraction, not a new fit; the
unconditioned collapse AUC reproduces part A at 0.547 exactly). Interpretation thresholds were
**predeclared before running**: a real residual requires matched AUC > 0.55 and permutation
p < 0.05 at the 0.25x-spread caliper or tighter, for absolute essentiality; PDAC-selective
(≈14 positives) was declared exploratory-only regardless of outcome.

**Absolute essentiality (n = 419, 31 essential) — the retraction deepens.** Degree-conditioned
concordance, tightening the degree caliper:

| caliper | matched AUC | unconditioned | pairs | perm p |
|---|---|---|---|---|
| 16.6 | 0.490 | 0.547 | 6,894 | 0.570 |
| 8.3 | 0.451 | 0.547 | 4,200 | 0.781 |
| 3.3 | 0.453 | 0.547 | 2,126 | 0.782 |

Once genes are matched on degree, collapse is **at chance or slightly below** (≤ 0.49), and never
significant. The unconditioned 0.547 was *entirely* degree. The tertile view confirms the
mechanism — collapse AUC rises monotonically with degree (low 0.404, mid 0.516, high 0.556),
tracking exactly where degree is itself informative (degree AUC 0.490 / 0.713 / 0.594). **For core
essentiality, collapse is degree in disguise** — a stronger and cleaner statement than the pooled
partial correlation.

**PDAC-selective essentiality (n = 419, 14 selective) — exploratory, and not claimed.** Here degree
is uninformative (low-tertile degree AUC 0.434), and the degree-matched collapse concordance sits
*above* its unconditioned value:

| caliper | matched AUC | unconditioned | pairs | perm p |
|---|---|---|---|---|
| 16.6 | 0.648 | 0.607 | 3,496 | 0.035 |
| 8.3 | 0.636 | 0.607 | 2,338 | 0.060 |
| 3.3 | 0.614 | 0.607 | 1,205 | 0.106 |

This is the one place collapse looks like it might carry degree-independent signal — and it is
exactly where a *disease-specific* dynamical model should add value if it adds any, since selective
(not core) vulnerability was the entire motivation for RAC. But it does **not** clear the bar to be
called a finding, and per the predeclared rule it is reported as a hypothesis only:

* significant (p 0.035) at the loosest caliper *only*; degrades to 0.106 under strict degree
  matching — the opposite of what a robust conditional effect does;
* does not survive even mild multiple-comparison correction (6 conditional tests; 0.035 x 6 ≈ 0.21);
* only 14 positives, and the eye-catching mid-tertile AUC of 0.829 rests on **4 genes**;
* the panel is known to contain ~7 strongly PDAC-selective TFs (§ single-cell power audit), so this
  contrast is underpowered by construction.

**What this adds.** The retraction is no longer a flat global null; it is *located*. Collapse
contributes nothing beyond degree for the essentiality that degree already predicts (core), and
shows only an underpowered, non-robust hint at the selective margin that degree does not predict.
The honest one-line summary: **collapse fails where degree already succeeds, and the one regime
where it might not fail is too thin to call.** The prospective test that would settle it — more
PDAC-selective dependencies than DepMap currently resolves — is the same data gap flagged
throughout this record, not a modelling choice.

### 15b. The selective hint, dissected: artifact, not signal — the retraction is now total and located

§15 left one thread open: for PDAC-*selective* essentiality the degree-matched concordance sat at
0.61-0.65, reported exploratory-only. Leaving a hint dangling is not depth; it is either a real
underpowered signal or an artifact, and those demand opposite follow-ups.
`scripts/selective_hint_dissection.py` -> `results/selective_hint_dissection.json` decides it on a
single fit (reproduces §15; selective unconditioned AUC 0.607), by a verdict rule fixed in code
before the run. **Verdict: LIKELY ARTIFACT.**

**Threshold-sensitive.** Degree-matched AUC at the predeclared 0.25x-spread caliper, across cuts:

| selective cut | n positive | matched AUC | perm p |
|---|---|---|---|
| 0.100 | 20 | 0.605 | 0.076 |
| 0.125 | 16 | 0.573 | 0.179 |
| 0.150 | 14 | 0.636 | 0.059 |
| 0.175 | 9 | 0.537 | 0.372 |
| 0.200 | 6 | 0.471 | 0.583 |

It swings from 0.64 to below 0.5, and is significant at **0 of 5** cuts. The p=0.035 in §15 came
from the *loosest* caliper (16.6); at the predeclared matching caliper nothing clears 0.05. The
"signal" existed at one label cut and one caliper.

**The biology falsifies it directly.** Naming the 14 positives at cut 0.15 with their collapse and
degree percentiles: only **1 of 14 (JUNB)** sits in the informative quadrant (high collapse, low/mid
degree) that a degree-independent signal requires. The rest are either high-degree (ZNF85, KLF5
[degree pct 1.00], SOX9, FOSL1, TCF7L2 — riding the confound through imperfect matching) or
low-collapse. Most tellingly, the two strongest selective dependencies in the panel are ranked
**near the bottom** by collapse:

| gene | selective essentiality | collapse percentile |
|---|---|---|
| KRAS | +1.474 (canonical PDAC driver) | **0.08** |
| MYC | +0.495 | 0.47 |

A model that captured PDAC-selective essentiality could not place KRAS at the 8th percentile. Even
KLF5 — this project's own recovered driver and the only curated PDAC-selective positive — owes its
position to degree (percentile 1.00), not collapse. Leave-one-positive-out (base 0.636, range
[0.598, 0.701]) shows removing a low-collapse gene *raises* the AUC, i.e. the number is diffuse
degree leakage, not a coherent per-gene effect.

**Consequence.** The §15 "thin hint at the selective margin" is **withdrawn**. There is no regime —
core or selective — where collapse adds discrimination over degree. The retraction, first stated
globally (§1) and then located to core essentiality (§15), is now **complete**: collapse is degree
in disguise everywhere it was tested, and the one place it appeared not to be was an artifact of a
threshold and imperfect matching. This is the strongest and cleanest form of the retraction, and it
was reached by dissecting the model's most flattering result rather than resting on it.

The bistable formulation's standing is therefore narrowed to exactly what survives: an *intervention
semantics* (clamp a node, observe attractor collapse) that is internally coherent but carries no
demonstrated predictive advantage over network degree on any essentiality endpoint available here.

---

## 16. Is the intervention gate's "safety" calibrated? Audited against local data — 11/12 corroborated

The review's standing objection to the signed intervention gate (§ intervention_gate) is that its
role assignments are a literature prior — hand-curated, not calibrated against anything measured.
`scripts/gate_role_data_audit.py` -> `results/gate_role_data_audit.json` tests every curated role
against a data-implied direction derived from independent local layers, and asks whether the two
agree.

**A metric bug, caught by sanity-checking against known biology.** The first version scored four
layers including `disease_log2fc` and protein-detection, and flagged the three canonical PDAC
tumour suppressors — SMAD4, CDKN2A, TP53 — as CONFLICT, i.e. "data says oncogenic." That is
impossible (SMAD4 is deleted in ~68% of tumours), so the audit was wrong, not the curation.
Diagnosis: `g.disease_log2fc` is a real TCGA-PAAD-vs-GTEx differential but CROSS-PLATFORM (TCGA RSEM
vs GTEx TPM), carrying a uniform +5.61 median batch offset that pins 91.8% of genes positive
(SMAD4 6.96, TGIF1 8.11); §16b shows even the batch-corrected differential cannot separate oncogene
from suppressor. (An earlier version of this section wrongly called the field an absolute
expression *level*; it is a batch-biased differential — the conclusion that it is unusable as a
direction is unchanged, the mechanism is corrected.) Protein-detection and unmethylated-default are
not directional either. The
one correctly-computed layer — copy number — had the right answer all along (SMAD4 deletion 0.77,
CDKN2A 0.90). The metric was corrected to score only genuinely directional layers: CNA (bidirectional,
primary) and promoter hypermethylation (one-sided, suppressor-only). This is the same failure the
rest of this document catalogues — a field used without verifying what it measures — reproduced in
an audit written to catch exactly that, and caught here by biology rather than by the number.

**Corrected result: 11 of 12 directional calls corroborated by independent copy number.**

| gate decision | genes | data-implied direction |
|---|---|---|
| ALLOW / repress (oncogenic) | SETDB1 +0.70, MYC +0.70, MYBL2 +0.60, E2F1 +0.55, HNF4G +0.50, KLF5 +0.38, FOSL1 +0.19 | all **amplified** — corroborated |
| BLOCK / tumour-suppressor | CDKN2A -0.70, SMAD4 -0.70, TP53 -0.70, FBXW7 -0.10 | all **deleted** — corroborated |
| CONFLICT | **GATA6 +0.69** | **amplified** — see below |
| inconclusive (CNA-invisible) | TGIF1 +0.04 | amp 0.15 ≈ del 0.13; a functional suppressor copy number cannot see |
| no local evidence | TP63 | absent from graph; protein-only, now unscored |
| quarantine-only (no admissible direction) | ARID1B -0.70, KMT2C +0.77, BRCA2 +0.33, ATM +0.23 | reported, not adjudicated |

Every one of the gate's seven **repression-allowed** oncogenes is independently corroborated as
amplified, and four of five tumour-suppressor blocks are corroborated as deleted. The gate's
directional calls are therefore calibrated against measured copy number, not merely asserted.

**The one conflict is the gate working, not failing.** GATA6 reads oncogenic by copy number (~30%
amplified in classical PDAC) yet is ACTIVATE-admissible / quarantined. That is not an error: GATA6
is an amplified *dependency* whose repression is nonetheless unsafe, because it maintains classical
identity and its loss drives the more aggressive basal state. GATA6 is precisely the gene where
"amplified and essential" and "safe to repress" diverge — the exact case a convergence score alone
gets wrong and the reason the signed gate exists. The audit surfacing GATA6 as the sole conflict,
with the gate's state-switch reasoning as the resolution, is the strongest single piece of evidence
that the gate encodes something a pure dependency ranking does not.

**Honest limits.** Copy number cannot adjudicate functional/expression-level suppressors (TGIF1,
whose loss accelerates PDAC without deletion) — reported inconclusive, not corroborated. TP63 is
absent from the graph and has no scored local evidence. And a true tumour-vs-normal expression
differential is not available locally as a clean contrast; adding GTEx-normalised expression would
let the audit adjudicate the functional suppressors it currently cannot.

### 16b. Why copy number, not expression, is the right instrument for direction (and a correction)

§16 excluded `disease_log2fc` and relied on copy number. The exclusion was right; the reason given
was wrong, and both are worth stating precisely because the distinction is the whole point.

**Correction.** `disease_log2fc` is not an absolute expression level. It is a genuine TCGA-PAAD
tumour vs GTEx normal-pancreas log2 differential — the same one `targeting/features.py` computes and
already flags as cross-platform. Across 1,666 matched genes its distribution is min -7.40, **median
+5.61**, max 13.03, with **91.8% of genes positive**: a large uniform batch offset from comparing
TCGA RSEM against GTEx TPM (GTEx pancreas TPMs are implausibly low — BRCA2 0.1, CDKN2A 0.2 — because
pancreatic RNA is enzyme-dominated and degradation-prone). The offset, not the biology, is what
pinned every gene positive.

**The deeper reason, which validates the CNA choice.** Remove the offset (subtract the genome-wide
median) and expression *still* does not separate oncogene from suppressor:

| set | genes | mean de-biased log2FC |
|---|---|---|
| oncogenes | MYC, KLF5, E2F1, MYBL2, SETDB1, FOSL1 | +1.32 |
| tumour suppressors | SMAD4, CDKN2A, TP53, TGIF1, FBXW7 | +1.23 |

Separation +0.09; Mann-Whitney (oncogene > suppressor) **p = 0.669**. SMAD4 (+1.35), TP53 (+1.36)
and TGIF1 (+2.50) — all suppressors — read *more* up than MYC (-0.21) and MYBL2 (-0.01). This is not
noise; it is biology: **PDAC tumour suppressors are inactivated by deletion and loss-of-function
mutation, not by transcriptional down-regulation**, so their mRNA is often normal or elevated even
as their function is lost. Expression differential is therefore structurally incapable of assigning
oncogene-vs-suppressor direction here, batch effect or no batch effect.

Copy number is the correct and, locally, the only valid directional instrument — amplification and
deletion *are* the mechanism of action in PDAC. §16's reliance on CNA was right for a stronger
reason than "the expression field looked buggy": it is right because the alternative cannot work.
The honest residual limit stands — CNA cannot see purely functional suppressors like TGIF1 — but no
locally available layer can, and claiming expression could would have been wrong.

---

## 17. The foundational premise: the system is not bistable (full addendum)

The retraction in §1/§15/§15b was phenomenological — collapse does not predict essentiality beyond
degree. A dynamical-systems audit now supplies the *mechanism*, and it is more fundamental than the
phenomenology: **the fitted RAC system is not bistable at its operating gain, and its "attractors"
are not attractors.** Full treatment in [docs/ADDENDUM_DYNAMICS.md]({{ '/addenda/dynamics/' | relative_url }});
scripts `dynamics_characterization.py` + `verify_dynamics_instability.py`; results in
`results/dynamics_characterization.json` + `results/verify_dynamics_instability.json`.

Four tests, three of them model-free (no eigenvalue), on the primary §15 fit:

| test | result |
|---|---|
| cell-state one-step residual (fixed-point quality) | median **0.729**, max 0.980 — not fixed points |
| convergence from 84 initial conditions (2000 iters) | **0 / 84** converge — no reachable stable attractor |
| perturbation growth over 20 steps | median **3.41×**, max 71× — expansive/unstable |
| spectral radius ρ (finite-diff = analytic) | **1.02–1.13** across equilibria — linearly unstable |
| bifurcation (gain sweep) | hysteresis only from gain **5.9**; operating 4.0 is below it |
| clamping reaches a dead basin | **0 / 10** nodes |

**Consequence.** `collapse_scores` calls `_settle` for 250 iterations on a map that never converges,
so it compares two *transient snapshots* with and without a node clamped — a **graph-influence
propagation** score, not a basin-transition score. A propagation score over a co-expression graph is
expected to track degree, which is exactly what §15 found. The mechanism explains the phenomenology.

**Retired:** "bistable attractor-control", "collapse to the dead attractor" as a mechanism,
"intervention semantics over a bistable system", and `control_design`'s "move the attractor"
(it moves a 250-step transient). **Honestly re-described:** a nonlinear graph-influence score over a
fitted co-expression-masked weight matrix, which does not beat degree. **Untouched:** the intervention
gate (§16, independent) and all data layers.

**A metric bug, caught by its guard.** The first eigenvalue estimate (ρ ≈ 2.6) was wrong — the
analytic Jacobian evaluated σ′ at `z` instead of at the map's argument `gain·z`. The
finite-difference cross-check, placed specifically to catch this, disagreed by 1.9 elementwise and
exposed it; corrected, analytic 1.019 = finite-diff 1.023. The conclusion never depended on the
eigenvalue (the convergence and perturbation tests are model-free), but the number was wrong and is
corrected rather than quietly dropped — the same discipline applied to the audit as to the code it
audits.

---

## 18. Can the method be rescued by operating in the bistable regime? No — tested, and the failure is structural

§17 identified that the operating gain 4.0 is below the bifurcation (~5.9). The obvious constructive
fix is "raise the gain." `scripts/gain_sweep_rescue.py` tests it directly — refit RAC at gains 4–8
and, at each, measure convergence, spectral radius, and collapse-vs-degree AUC (against the fixed
§15 degree reference 0.629). Interpretation predeclared: rescue = a convergent/stable regime **and**
collapse > degree.

| gain | converged | ρ | collapse AUC | ΔAUC vs degree |
|---|---|---|---|---|
| 4.0 | 0.00 | 1.02 | 0.606 | −0.023 |
| 5.0 | 0.00 | 1.26 | 0.572 | −0.057 |
| 6.0 | 0.00 | 1.34 | 0.586 | −0.043 |
| 7.0 | 0.00 | 0.96 | 0.674 | +0.045 (isolated) |
| 8.0 | 0.00 | 0.92 | 0.615 | −0.014 |

**Neither axis rescues.** Convergence is 0.00 at *every* gain — even where ρ < 1 (gains 7, 8) the
map does not settle, so ρ is measured at a transient snapshot and certifies no fixed point. And
collapse never robustly beats degree: four of five gains negative, the lone positive (gain 7,
+0.045) isolated, non-convergent, and inside §15's ΔAUC noise band (±0.11) — reported, not claimed,
because selecting it would be the one-in-five outcome-selection §15 exists to prevent.

**This makes the §15 retraction structural.** Dynamics propagate only what the graph encodes; §15/§15b
showed essentiality is not in the co-expression graph beyond degree (partial ρ = 0.028, KRAS 8th
percentile), so no reparameterisation over that graph can manufacture it. RAC is not one
hyperparameter from working; a real rebuild would have to change what the graph encodes (directed/
causal edges, perturbation data), not how the dynamics run over it. It also corrects §17's own
constructive suggestion: raising the gain is necessary but not sufficient — the fit must be made to
produce true, stable fixed points first. Full detail in [docs/ADDENDUM_DYNAMICS.md]({{ '/addenda/dynamics/' | relative_url }}) §7.

---

## 19. RAC v2 rebuild attempt — substrate-first, stopped at the gate (full addendum)

§18 concluded RAC's failure is structural in the substrate. The constructive response is a rebuild,
executed in the order the diagnosis dictates: **fix the substrate first, prove it carries signal,
and only then build dynamics.** Full write-up in [docs/ADDENDUM_RAC_V2.md]({{ '/addenda/rac-v2/' | relative_url }});
scripts `build_directed_grn.py` + `substrate_signal_test.py`.

**Phase 1 — a directed substrate.** The current graph is symmetric co-expression (degree ≈ hub-ness
≈ essentiality tautologically). Built a directed TF→target motif-GRN instead: edge i→j = best JASPAR
PWM hit of TF i in gene j's hg38 promoter. 422 nodes, 205 regulators with PWMs, 11,331 edges at
score ≥ 0.9. Leakage-free vs DepMap CRISPR.

**Phase 2 — the gate (predeclared): does the directed topology beat degree?** Against DepMap absolute
essentiality (co-expression degree AUC 0.629 as baseline), every directed property fails:

| property | raw AUC | matched AUC | partial p vs degree |
|---|---|---|---|
| out-strength / out-degree | 0.39 / 0.43 | 0.39 / 0.43 | 0.35 / 0.29 |
| in-strength / in-degree | 0.47 / 0.51 | 0.48 / 0.50 | 0.52 / 0.53 |
| directed PageRank | 0.48 | 0.44 | 0.86 |
| HITS hub / authority | 0.43 / 0.54 | 0.43 / 0.54 | 0.28 / 0.74 |

Not one beats degree (raw all < 0.629; matched all < 0.55; partial all n.s.). Threshold-free weighted
strengths fail too, so it is not a cutoff artifact. The "master regulator drives many genes"
hypothesis is refuted with the sign — out-degree is *anti*-associated with essentiality (0.39).

**Decision: STOP; the failure is confirmed across two independent topologies.** Both undirected
co-expression and directed motif-regulatory graphs fail to encode PDAC-TF essentiality beyond node
degree. The signal is not in the edge directions or a smarter centrality — it is not in the graph
topology at all. Genuine progress needs different *information* (perturbation data as input, a causal
network learned from interventions, a larger panel), not a reparameterisation of RAC. Building
bistable dynamics on this substrate would repeat the diagnosed error one level up, so the rebuild
stops here — the correct, honest outcome. The intervention gate (§16) and data layers are unaffected.

---

## 20. The achievable ceiling (RAC v2 Phase 4) — and the one weak positive

With graph topology ruled out (§19), the quantitative question remained: is degree the ceiling for
*any* available feature? A nested-CV supervised model (logistic + GBM, 5 seeds, fixed defaults) was
fit on every leakage-free feature (degree, directed-GRN properties, CNA, methylation, expression,
disease log2FC). Full detail: [docs/ADDENDUM_RAC_V2.md]({{ '/addenda/rac-v2/' | relative_url }}) §8.

**Absolute essentiality:** full model 0.85 vs degree 0.62 — but the gain is carried by `expr_mean_raw`
(univariate 0.809), the DepMap tautology that an unexpressed gene has Chronos ≈ 0. Real, not
actionable (you cannot drug "being expressed"); topology and CNA add nothing (all ≤ degree or ≈ 0.52).

**Selective essentiality (the endpoint that matters):** degree is *anti*-predictive (0.424). The
multi-omic model reaches 0.651, and it was held to the §15b bar that killed the collapse hint:
matched permutation null p = 0.030 (I first ran the null on the wrong 14-feature set — 0.516,
chance — and corrected it to match the 0.651); threshold sweep 0.10–0.20 significant at the two
best-powered cuts (p = 0.010, 0.020 at n = 20, 16). **Unlike §15b, this is not a threshold artifact:**
the observed AUC is stable at ~0.65–0.70 and sits ~0.15–0.20 above the null mean at *every* cut;
significance fades at the high cuts only because positives drop to 6–9 (a power effect, not
effect-size collapse). The script's naive "2/5 significant → artifact" auto-verdict is overruled by
reading the effect-size stability.

**Honest verdict:** a **weak, real, underpowered** PDAC-selective signal that degree, topology, and
RAC dynamics all miss — recoverable by ordinary supervised learning over the assembled multi-omic
data, not by attractor dynamics. It rests on ~20 genes, is marginal after the five cuts, and leans on
expression; a hypothesis for prospective testing on a larger selective panel, not an established
predictor. **The value across this project is the data assembly and the intervention gate (§16), not
the attractor model** — which is retracted (§15), retired (§17), and unrescuable by gain (§18) or
substrate (§19).

---

## 21. Phase 5 — the last positive is an expression confound, and the model is worse than its own feature

RAC v2 §8.3 recorded the investigation's single surviving positive (a supervised multi-omic model
recovering PDAC-selective essentiality, CV-AUC 0.651/0.683 where degree is anti-predictive at 0.424)
and named, without testing, the danger that it "leans on raw expression … not proven free of a
measurement confound." §9 of [docs/ADDENDUM_RAC_V2.md]({{ '/addenda/rac-v2/' | relative_url }}) is that test
(`scripts/selective_confound_test.py`). The claim does not survive intact.

**Protocol reproduced first:** full model returns 0.6510 logistic / 0.6833 GBM against the expected
0.651/0.683 — so all comparisons are valid. (Guard added because a previous null was run on a
mismatched feature set.)

**A single expression feature beats the whole model.** Univariate CV-AUC on the *selective* endpoint
— never previously measured; the earlier artifact check covered only absolute essentiality:

| feature | univariate CV-AUC |
|---|---|
| **expr_mean_raw** | **0.777** |
| disease_log2fc_LEVEL | 0.610 |
| expr_var_norm | 0.604 |
| in_strength | 0.597 |
| expr_pdac_minus_other_DIFFERENTIAL | 0.557 |
| coexpr_degree | 0.424 |
| pagerank | 0.363 |

`expr_mean_raw` alone (**0.777**) exceeds the full 16-feature model (0.651 / 0.683). The multi-omic
integration does not add value here — it dilutes one strong column.

**My predeclared hypothesis was refuted, informatively.** I predicted the PDAC-vs-other expression
*differential* would carry the tautology ("selectively expressed ⇒ selectively essential", threshold
≥0.62). It scores only **0.557**, and adding it to the model *hurts* (0.651 → 0.634). The real
confound is **absolute PDAC expression**: it predicts absolute essentiality at 0.809 and selective at
0.777, because expression drives the PDAC arm of `sel = −(pdac − other)`. Expression confounds the
minuend, not the difference — a reader checking only the differential would have wrongly cleared it.

**Ablation splits by model class, and is reported split rather than collapsed:**

| model | with expression | without expression (11 feats) |
|---|---|---|
| logistic | 0.651 ± 0.010 | **0.450 ± 0.027** |
| GBM | 0.683 ± 0.031 | **0.625 ± 0.043** |

The linear signal is entirely expression (0.450 is below chance). A nonlinear model retains 0.625 on
topology + CNA + methylation — reported as an **underpowered hypothesis** (14 positives, ~2.8 per
fold, widest variance in the study). What it likely learns is visible in the univariate table:
pagerank 0.363, out_degree 0.376, hits_hub 0.380 are strongly *anti*-predictive, i.e. **selective
dependencies sit at the graph periphery, not at hubs** — consistent with §15b ranking KRAS at the
8th percentile. Disclosed limitation: the ablation removes expression *levels* but `coexpr_degree`/
`eigenvector` still derive from co-expression, so it is a partial ablation.

**Corrected position.** Not supported: "a supervised *multi-omic* model recovers selective signal."
Supported with a heavy caveat: PDAC-selective essentiality is predicted by mean PDAC expression
(0.777) through a substantially near-definitional channel (same cell lines; a gene cannot be
essential where it is not expressed) — expected biology, not a discovery, requiring neither a network
model nor multi-omic integration. What remains genuinely defensible from this project is the **data
assembly** and the **data-calibrated intervention gate** (§16), not the attractor model and not the
supervised ceiling.

---

## 22. Phase 6 — the graph-peripheral hypothesis is refuted, and the screen that produced it was unsound

§21 closed the supervised ceiling as expression-carried but opened one new observation: several
centrality features scored below 0.5 on the selective endpoint, from which §9.3 inferred *"inverted
predictiveness is predictiveness — selective dependencies sit at the graph periphery."* That is now
withdrawn. Full treatment: [docs/ADDENDUM_RAC_V2.md]({{ '/addenda/rac-v2/' | relative_url }}) §10.

**Two demonstrated errors in the instrument, not just the hypothesis:**

1. **Logistic CV-AUC is exactly invariant to negating a feature** — fitting `−x` flips the coefficient
   and leaves predictions identical (verified: `AUC(x) = AUC(−x) = 0.423986`). The "peripherality =
   −centrality" composite was a **no-op** re-test of centrality.
2. **A fitted OOF AUC below 0.5 does not mean inverted signal at 14 positives.** On synthetic data
   with a genuinely *positive* association (model-free AUC 0.562), the 5-fold logistic OOF AUC is
   **0.424** — dragged below chance by ~2.8 positives per fold. Sub-0.5 fitted AUCs carry no
   directional information.

**Model-free rank screen — six features flip direction.** out_degree 0.376→**0.558**, hits_hub
0.380→**0.556**, coexpr_degree 0.424→**0.527**, expr_var_raw 0.412→**0.549**, expr_var_norm
0.604→**0.373**, in_strength 0.597→**0.378**. The centralities behind the peripherality claim land
*slightly above* 0.5 — the opposite direction — and disagree with each other. That is noise.

**Nothing survives correction.** Over 17 continuous Spearman tests, BH gives **zero survivors**; the
best is expr_mean_raw at p = 0.024 → **q = 0.402**. At this sample size the selective endpoint
supports no feature-level claim that survives multiple testing. It also sharpens §21: expr_mean_raw
is strong on the dichotomised tail (rank AUC 0.794) but only weakly monotone overall (ρ = 0.111) —
it separates the extreme selective genes rather than tracking selectivity, exactly as a confound
would.

**Transferable lesson:** for univariate screening with a small positive class, use a model-free rank
statistic. A fitted CV model cannot express direction (it learns its own sign) and is biased below
chance by fold instability — and reading direction off it produced a confident, plausible, wrong
story about network topology.

---

## 23. Phase 7 — the detection floor, and two corrections it forces

§22 ended on "nothing survives BH", which is ambiguous between *no signal exists* and *none was
detectable*. `scripts/selective_power_floor.py` quantifies the floor (Mann-Whitney power; for target
rank-AUC `a`, normal scores separated by `d = √2·Φ⁻¹(a)`; BH bounded conservatively by α/17). Full
treatment: [docs/ADDENDUM_RAC_V2.md]({{ '/addenda/rac-v2/' | relative_url }}) §11.

**Minimum detectable rank-AUC at the actual design (14 positives, BH): 0.771.** Power at 14
positives is **7%** for a 0.65 effect, 41% for 0.70. Positives required for the observed effects:
expr_mean_raw (0.794) → 14 (sufficient); expression differential (0.611) → 120; best centrality
(0.563) → >400.

**Correction 1 — §22's "nothing survives BH" was over-generalised.** That holds for the *continuous*
Spearman tests but not the *dichotomised* one. Recomputing BH over dichotomised Mann-Whitney
p-values: **expr_mean_raw p = 0.00018, q = 0.0030 — survives robustly**; nothing else does (next
q = 0.374). (My first pass at this table had a BH bug — a forward running minimum instead of the
reverse cumulative minimum — which reported q = 0.003 for p = 0.105. Caught because a q three orders
below its own p is impossible.)

**The synthesis is sharper than any single phase.** Expression *separates the extreme selective
tail* robustly (AUC 0.794, q = 0.0030, and adequately powered) but *does not grade selectivity*
(continuous ρ = 0.111, q = 0.402). That combination is the signature of a **threshold-like
confound** — whether a gene is expressed in PDAC at all — rather than a graded biological predictor,
which is precisely the near-definitional channel §21 identified. Everything else was undetectable by
construction.

**Correction 2 — the §22 "refutation" of the graph-peripheral hypothesis is calibrated down.** With
7–19% power for effects of 0.60–0.65, non-significance proves nothing about centrality effects in
either direction. The refutation rests solely on its other leg — model-free point estimates
*contradicting* the hypothesis (centralities slightly above 0.5, opposite to "peripheral", and
mutually inconsistent). Calibrated verdict: **unsupported, point estimates pointing the wrong way,
underpowered to settle definitively** — weaker than "refuted", and what the data supports.

**Design specification for any future attempt** (80% power, BH over ~17 features): rank-AUC 0.75
needs ~20 positives; 0.70 needs ~30; 0.65 needs ~50–60; 0.60 needs ~100+. This panel has **14**, and
only ~7 of 1,164 TFs are strongly PDAC-selective — so this is not fixable by better statistics on
DepMap, it needs a larger or differently-constituted selective-dependency panel.

---

## 24. The exact Doench-2016 CFD matrix: `undetermined` resolved, and my own prediction refuted

The specificity verdict on the four gated guides has sat at `undetermined` since §14b, blocked on one
thing: the repository's `cfd_style_score` was a **position-granular** approximation (Hsu-2013 weights
standing in for the Doench-2016 nucleotide-**pair** matrix), and I declined to reconstruct the real
coefficients from memory or to unpickle CRISPOR's `.pkl` (arbitrary code execution). The matrix has
now been obtained as **plain text** and the verdict is resolved.

**Provenance.** `data/raw/doench2016-cfd/` + `data/manifests/doench2016-cfd.json` (sha256, `REAL`):
240 nucleotide-pair mismatch weights + 16 PAM weights, redistributed as plain text by the
Bioconductor package `crisprScore` (crisprVerse/crisprScore, `inst/cfd_cas9/`) from Doench et al.
2016, *Nat Biotechnol* 34:184–191. Implementation: `src/pdac_circuit/grna/cfd_doench.py`.

**Two verification steps, because a silently-inverted matrix would corrupt every score.**
crisprScore keys its table `{spacer}{protospacer}{position}` in DNA space with **no
reverse-complement** — it has already absorbed the revcomp the original `cfd_score_calculator.py`
applies. Using the original convention against this file would invert it. The order was fixed by
arithmetic: the package's own test vector expects **0.765** for a spacer-A/protospacer-G mismatch at
position 20, matching `AG20 = 0.7647`, *not* `GA20 = 0.9375`. The loader then reproduces **all seven
published Cas9 test vectors exactly** and raises if it cannot, so a corrupted matrix fails loudly.
(Independent bonus check: the downloaded Hsu-2013 MIT weights match this repo's hardcoded
`HSU_WEIGHTS` exactly.)

**Resolved verdict — all four guides FAIL the pre-registered gate (cfd_specificity ≥ 0.5):**

| gene | off-targets ≤4mm | ≤2mm | ≤3mm | ≤4mm | gate |
|---|---|---|---|---|---|
| SETDB1 | 206 | 0.667 | 0.317 | **0.045** | FAIL |
| MYBL2 | 75 | 1.000 | 0.469 | **0.082** | FAIL |
| E2F1 | 68 | 1.000 | 0.447 | **0.126** | FAIL |
| FOSL1 | 35 | 1.000 | 0.700 | **0.308** | FAIL |

**My prediction was wrong, and in the consequential direction.** Across §14/§14b I repeatedly wrote
that the position-granular figures were "likely **pessimistic**" and that the truth lay somewhere
better. The exact matrix mostly made things **worse**, and the error is **non-monotone** — it went
both ways:

| gene | cutoff | approximation | exact | change |
|---|---|---|---|---|
| SETDB1 | ≤2mm | 0.862 | **0.667** | −0.195 |
| MYBL2 | ≤3mm | 0.659 | **0.469** | −0.190 (PASS → **FAIL**) |
| E2F1 | ≤3mm | 0.510 | **0.447** | −0.063 (PASS → **FAIL**) |
| FOSL1 | ≤4mm | 0.218 | **0.308** | +0.090 |

So §14b's headline — *"at ≤3 mm, MYBL2, FOSL1 and E2F1 clear the gate; only SETDB1 fails robustly"* —
is **withdrawn**. At ≤3 mm only **FOSL1** passes (0.700). The approximation was optimistic exactly
where the earlier conclusion leaned on it.

**Characterised worst sites (exact CFD):**

| gene | locus | mm | CFD |
|---|---|---|---|
| E2F1 | chr11:37,152,430 | 4 | **0.603** |
| SETDB1 | chr19:53,891,820 | **2** | **0.500** |
| MYBL2 | chr17:39,401,140 | 3 | 0.476 |
| FOSL1 | chr17:83,101,244 | 3 | 0.220 |

FOSL1's previously-alarming "CFD 0.921" site drops to **0.220** under the exact matrix — the
approximation badly over-scored that one site while under-scoring others, which is precisely why a
position-granular proxy could not settle this question either way.

**Standing position.** No guide in this repository is orderable on specificity grounds — now stated
with a published, validated scorer rather than a proxy, and no longer hedged as `undetermined`.
Every guide remains uniquely placed (exactly one perfect genomic match each). All future scores
carry a `cfd_scorer` provenance field (`doench2016_exact`) so no result is ambiguous about which
scorer produced it.

---

## 25. Fold-change chromatin: the H3K27ac enrichment SURVIVES — the first claim in this record to do so

The README has carried "PANC-1 chromatin concordance — directional only; must be recomputed on
fold-change tracks" since the original review. The tracks are downloaded and the recompute is done.
Scripts: `fetch_foldchange_tracks.py`, `pdac_residual_foldchange.py`. Results:
`results/pdac_residual_foldchange_{H3K27ac,ATAC-seq}.json`. Provenance:
`data/manifests/encode-foldchange.json` (4 artifacts, sha256, REAL).

**Why the old number was untrustworthy.** It used ENCODE **signal p-value** tracks, which conflate
sequencing depth with enrichment — a deeper library yields larger p-values for the same true
enrichment. Fold-change-over-control is depth-normalised against the matched input. Files were
selected by matching the `derived_from` **processing run** of the p-value tracks they replace
(Panc1 H3K27ac ENCFF528UFR to ENCFF047WWJ, both from ENCFF384KMQ+ENCFF675MQQ; Panc1 ATAC
ENCFF055ZEE to ENCFF174PXJ, both from ENCFF836WDC), so old-vs-new changes normalisation **only**.
ENCSR000EXK contains a second run whose fold-change file (ENCFF240BXE) derives from different
alignments; picking it would have changed two things at once.

**Three matched controls, because the original test had a worse flaw than its normalisation.**
"Targets vs all background" risks measuring expression, not disease — the confound §21 showed
carries the supervised selective signal. Each target was paired with up to 3 background genes
(0.25 SD caliper) on three orthogonal variables.

### H3K27ac — survives everything (n_targets = 20, n_loci = 1,675)

| contrast | matched score (tgt vs bg) | targets | background | MWU p |
|---|---|---|---|---|
| all background | — | **+0.919** | −0.091 | **0.0022** |
| absolute PDAC expression | 2.232 vs 2.230 | +0.919 | +0.105 | **0.017** |
| **`disease_log2fc`** (circularity) | 6.867 vs 6.75 | +0.919 | +0.022 | **0.025** |
| **co-expression degree** (hub-ness) | 136.8 vs 136.75 | +0.919 | −0.017 | **0.010** |

The two controls that mattered most both hold. **Circularity:** RAC selected targets partly on
`disease_log2fc` (a TCGA-vs-GTEx differential) while this residual is a Panc1-vs-healthy-pancreas
contrast — both "PDAC vs normal pancreas", so a disease-up gene would have disease-up promoter
acetylation almost by construction. Matching on absolute expression does *not* control for that (it
matches level, not change); matching on `disease_log2fc` does, and the effect holds. **Hub-ness:**
§15 established collapse is degree in disguise, so "hub TFs sit on PDAC-gained enhancers" was the
most likely mundane explanation. With degree matched to 136.8 vs 136.75 the effect is if anything
*stronger* (p = 0.010). It is not a hub artifact.

### ATAC-seq — does NOT replicate

Primary contrast **+0.421 vs +0.177, p = 0.074 — not significant**. Its matched controls are
nominally significant (0.0045–0.047), and they are **not** used to claim a positive: significant
matched subsets do not rescue a failed primary contrast, because that is choosing whichever
comparison happens to work. My first verdict function evaluated only the matched controls and
reported "SURVIVES all matched controls" for ATAC; the primary contrast now gates the verdict, and
ATAC reads "does not replicate". Same failure mode as elsewhere in this record — an auto-verdict
that did not check the thing that mattered.

### What this does and does not mean

**Does:** the 20 RAC-surfaced genes sit on PDAC-gained H3K27ac beyond expression level, beyond the
disease-expression change they were selected on, and beyond hub-ness. Magnitude drops from the
published signal-p-value figure (+1.596 to +0.919) as depth normalisation should, while direction
and significance hold. After a session in which every other claim was retracted, retired, shown
unrescuable, or exposed as confounded, this is the **first result to survive a deliberate attempt to
kill it**.

**Does not:** resurrect RAC. The targets come from a model whose essentiality claim is retracted
(§15), whose bistable framing is retired (§17), and which is unrescuable by gain (§18) or substrate
(§19). This says the *gene set* has a chromatin property; it says nothing about collapse predicting
essentiality.

**Standing caveats, none of which the result removes:** PANC-1 is an **intermediate** substrate (§9
— below panel mean on both Moffitt programmes, representing neither subtype); the fold-change corpus
has **one** healthy track per mark versus six averaged in the p-value run; n = 20 targets; and the
weakest control would not clear a Bonferroni across the three, so this is a robustness demonstration
rather than an independently-powered confirmation. **ATAC not replicating means the effect is
mark-specific** and should be described as an H3K27ac observation, not a general chromatin claim.

---

## 26. Stress-testing the one survivor: the H3K27ac result holds

§25's H3K27ac enrichment is the only claim in this record to survive a deliberate attempt to kill
it, which is precisely why it should face the treatment that killed the others. §15b's lesson was
that a result can clear its primary test and still be carried by one or two genes, or exist only at
one arbitrary threshold. Nothing in §25 ruled that out. `scripts/h3k27ac_fragility.py` →
`results/h3k27ac_fragility.json`.

**A. Set-level permutation — the correct null, and it is stricter than the one I reported.**
Mann-Whitney treats genes as exchangeable individuals, but the claim is about a **set of 20**.
Drawing 20 random background genes 20,000 times: observed set mean **+0.919** against a null mean of
−0.093 and a null 95th percentile of **+0.461**. **p = 0.00155** — more significant than the
gene-wise MWU (0.0022), not less. The §25 p-value was not significance borrowed from a false
independence assumption.

**B. Leave-one-target-out — not one gene.** Dropping each target in turn, the *worst* case is
p = 0.0058, when removing **HOXA3** (the single largest contributor at +5.88). Even excising the
biggest driver the result stays an order of magnitude below 0.05.

**C. Per-gene distribution, including a coherence check that passes.** Top: HOXA3 +5.88, FOSL1
+2.77, MYBL2 +2.19, SMAD3 +2.12. Bottom: FAM83A −0.56, ZNF85 −0.71, **GATA6 −2.24**. That GATA6 is
the most *negative* target is an internal consistency check nobody designed: GATA6 is the classical
identity factor, and §9 established PANC-1 sits **below** the panel mean on the classical programme.
A classical-identity enhancer should be *less* acetylated in a non-classical line, and it is. The
one gene whose direction was predictable from independent evidence points the right way.

**D. Bootstrap 95% CI on the target mean: [0.244, 1.679].** Excludes zero, and is **wide** — the
effect is real but imprecisely estimated at n = 20. Reported as an interval rather than a point
because the point overstates what 20 genes can pin down.

**E. Caliper sensitivity — flat, which is the §15b discriminator.** All three matched controls at
calipers 0.10 / 0.25 / 0.50 SD:

| caliper | expression | `disease_log2fc` | degree |
|---|---|---|---|
| 0.10 | 0.017 | 0.030 | 0.010 |
| 0.25 | 0.017 | 0.025 | 0.010 |
| 0.50 | 0.017 | 0.019 | 0.010 |

Every cell significant, and essentially flat across a 5× caliper range. This is the exact contrast
with §15b, where the collapse hint bounced (0.605 / 0.573 / 0.636 / 0.537 / 0.471) and was correctly
called an artifact. Stability across the free parameter is what separates the two.

**Verdict: ROBUST.** Survives the set-level permutation, leave-one-out, and every caliper.

**What still does not change.** PANC-1 remains an **intermediate** substrate (§9); one healthy
fold-change track versus six averaged; n = 20; **ATAC does not replicate** (primary p = 0.074), so
this is an H3K27ac-specific observation; and it does **not** resurrect RAC — the essentiality claim
stays retracted (§15), the bistable framing retired (§17), the rebuild unrescuable (§18/§19). The
defensible statement is unchanged in scope and now much better supported: *the 20 RAC-surfaced genes
sit on PDAC-gained H3K27ac beyond expression, beyond the disease-expression change they were
selected on, and beyond hub-ness — robustly, and not because of any single gene.*

---

## 27. Window sensitivity and per-locus detail — robust, with two corrections to how §25/§26 read

§26 varied caliper, gene membership and null model, but never the parameter baked into the
measurement itself: the **TSS ± 2 kb window**, inherited from `pdac_disease_residual.py` and never
justified. A result living at one arbitrary threshold is the §15b artifact signature, so this was
the last open methodological hole. `scripts/h3k27ac_window_and_loci.py` →
`results/h3k27ac_window_and_loci.json`.

### Significant at 6/6 widths — but read the contrast, not the target mean

| window | target log2 | target fold | background log2 | **contrast** | MWU p | set-perm p |
|---|---|---|---|---|---|---|
| ±500 bp | +1.036 | **2.05×** | +0.231 | +0.805 | 0.032 | 0.040 |
| ±1 kb | +0.787 | 1.73× | +0.108 | +0.680 | 0.018 | 0.041 |
| **±2 kb (published)** | +0.919 | 1.89× | −0.091 | **+1.011** | **0.0022** | **0.0022** |
| ±5 kb | +0.449 | 1.36× | −0.316 | +0.765 | 0.0047 | 0.0056 |
| ±10 kb | +0.231 | 1.17× | −0.422 | +0.653 | 0.0070 | 0.0093 |
| ±25 kb | +0.016 | **1.01×** | −0.533 | +0.549 | 0.0074 | 0.015 |

**Correction 1 — the effect is promoter-local, and at wide windows the targets are not enriched at
all.** The target–background *contrast* is stable (+0.55 to +1.01), which is what drives significance
everywhere. But the target's **absolute** enrichment collapses from 2.05× at ±500 bp to **1.01× at
±25 kb** — no gain whatsoever. Significance at wide windows comes from the *background* becoming
depleted (−0.533), not from targets staying high. The honest two-part statement: targets **gain**
acetylation at promoter scale, and **retain** it at domain scale where other TF loci lose it.
Describing this as a broad "PDAC-gained enhancer domain" would be wrong.

**Disclosed, not buried: ±2 kb is the most favourable of the six windows** (largest contrast,
smallest p). It was inherited rather than chosen after seeing results, and all six widths are
significant, so this is not a lucky cut — but the published number sits at the optimum and should be
read that way.

### Correction 2 — the top locus is substantially a pseudocount artifact

Per-locus inspection at ±2 kb (linear PDAC/healthy ratio):

| gene | locus | PDAC FC | healthy FC | ratio |
|---|---|---|---|---|
| **HOXA3** | chr7:27,150,583-27,154,583 | 5.772 | **0.000** | **58.7×** |
| FOSL1 | chr11:65,898,573-65,902,573 | 20.04 | 2.85 | 6.83× |
| MYBL2 | chr20:43,665,019-43,669,019 | 5.55 | 1.14 | 4.57× |
| SMAD3 | chr15:67,061,763-67,065,763 | 17.50 | 3.95 | 4.35× |
| ZNF528 | chr19:52,395,849-52,399,849 | 6.56 | 1.58 | 3.95× |
| E2F1 | chr20:33,684,385-33,688,385 | 5.96 | 1.65 | 3.46× |
| … | | | | |
| GATA6 | | | | **0.21×** |

**HOXA3's healthy fold-change is exactly 0.000**, so its 58.7× ratio is `log2((5.772+0.1)/(0+0.1))`
— a number set entirely by the arbitrary pseudocount `PSEUDO = 0.1`, not by data. Change the
pseudocount and HOXA3's contribution changes arbitrarily. It was the single largest contributor to
the headline mean, and it is the *only* target with near-zero healthy signal (the other 19 have real
signal on both sides). Whether healthy is truly unacetylated there or the track simply lacks
coverage cannot be distinguished from a fold-change file.

**Effect size restated without it:**

| statistic | value | fold |
|---|---|---|
| mean (published, with HOXA3) | +0.919 | 1.89× |
| **mean excluding HOXA3** | **+0.659** | **1.58×** |
| **median (outlier-robust)** | **+0.852** | **1.81×** |

§26's leave-one-out already established the result survives HOXA3's removal (MWU p = 0.0058,
set-permutation p = 0.017), so the finding stands — but the headline **1.89× overstates it**, and
**1.6–1.8× is the defensible range**. The median, which was always robust to this, sits at 1.81×.

### Net effect

The finding is **more robust than §25 claimed on the window axis** (6/6 widths, contrast stable
across a 50× range of widths) and **smaller than §25 claimed on effect size** (1.58–1.81×, not
1.89×). Both corrections come from looking at the per-locus data rather than the summary statistic —
the aggregate was hiding one degenerate denominator and one qualitative change in what "enrichment"
meant as the window widened. Everything else stands: PANC-1 is an intermediate substrate (§9), one
healthy track, n = 20, ATAC does not replicate, and this does not resurrect RAC.

---

## 28. Pseudocount sensitivity — the last unvaried constant, and what it settles about effect size

§27 showed the pseudocount is not cosmetic: HOXA3's healthy fold-change is exactly 0.000, so its
entire 58.7× ratio is `log2(5.872 / 0.1)` — manufactured by the constant. Having found that
`PSEUDO = 0.1` decides one locus outright, leaving the aggregate's dependence on it untested would
repeat the omission §27 had just corrected for the window. `scripts/h3k27ac_pseudocount.py` →
`results/h3k27ac_pseudocount.json`. Raw per-locus fold-changes are read once and reused, so the
sweep varies exactly one thing.

| pseudocount | mean log2 | fold | median log2 | mean excl. zero-denominator | MWU p | set-perm p |
|---|---|---|---|---|---|---|
| 0.01 | +1.101 | 2.15× | +0.891 | +0.676 | 0.0021 | 0.0065 |
| 0.05 | +0.978 | 1.97× | +0.877 | +0.668 | 0.0022 | 0.0023 |
| **0.1 (published)** | +0.919 | 1.89× | +0.852 | +0.658 | 0.0022 | 0.0022 |
| 0.5 | +0.747 | 1.68× | +0.707 | +0.595 | 0.0023 | 0.0014 |
| 1.0 | +0.648 | 1.57× | +0.636 | +0.536 | 0.0026 | 0.0028 |
| 2.0 | +0.532 | 1.45× | +0.564 | +0.457 | 0.0032 | 0.0033 |

Only **one target** (HOXA3) has zero healthy signal; 67 of 1,675 loci overall.

**Significance is essentially immune to the constant — and there is a reason, not a coincidence.**
MWU p stays in 0.0021–0.0032 and the set-level permutation in 0.0014–0.0065 across a **200× range**
of pseudocounts. Both are **rank-based**, and changing the pseudocount is very nearly a monotone
transform of every locus's ratio, so the gene ordering barely moves. This retrospectively validates
the test structure: the rank test carries the claim and is unaffected, while the mean carries the
effect size and is not. Had significance been reported via a mean-difference t-test it would have
drifted with an arbitrary constant.

**Effect size is genuinely parameter-dependent, and this is the honest bound.** Across the two
swept parameters the target enrichment spans:

| parameter | range tested | fold-change span |
|---|---|---|
| window | ±500 bp – ±25 kb | 1.01× – 2.05× |
| pseudocount | 0.01 – 2.0 | 1.45× – 2.15× |

**All 12 settings are significant; not one gives a stable magnitude.** The estimators that resist
the constant are the median (+0.564…+0.891) and the zero-denominator-excluded mean
(+0.457…+0.676) — both far tighter than the raw mean's +0.532…+1.101. At published settings those
give **1.58–1.81×**, consistent with §27.

**Final characterisation of the one surviving claim.** The 20 RAC-surfaced genes sit on PDAC-gained
H3K27ac at their promoters. The *existence* of the effect is robust: significant under every matched
control (§25), the set-level null and leave-one-out (§26), all six windows (§27), and all six
pseudocounts (§28) — 12 parameter settings, no exceptions. The *magnitude* is **not** well
determined: roughly **1.5–1.8×** on the estimators that resist arbitrary constants, with a plausible
range of 1.0–2.2× depending on choices no data constrains. Reporting a single headline number for
this effect would misrepresent it; the claim is "real, promoter-local, and modest", not "1.89×".

Unchanged: PANC-1 is an intermediate substrate (§9); one healthy track; n = 20; **ATAC does not
replicate**; and none of this resurrects RAC.
