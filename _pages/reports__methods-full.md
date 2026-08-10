---
layout: default
title: "Methods in full"
subtitle: "Complete methodological detail for every module and model."
description: "Complete methodological detail for every module and model."
permalink: /reports/methods-full/
group: reports
order: 3
---

> ### ⚠️ STATUS: several results in this document are RETRACTED or SUPERSEDED
>
> This file predates the review arc recorded in **[REVIEW_RESPONSE.md]({{ '/reports/review/' | relative_url }})** (§1–§28)
> and the addenda **[docs/ADDENDUM_DYNAMICS.md]({{ '/addenda/dynamics/' | relative_url }})**,
> **[docs/ADDENDUM_RAC_V2.md]({{ '/addenda/rac-v2/' | relative_url }})**,
> **[docs/ADDENDUM_CHROMATIN.md]({{ '/addenda/chromatin/' | relative_url }})**. In particular:
>
> - **attractor-collapse predicts essentiality — RETRACTED** (no gain over degree; §1/§15/§15b)
> - **the bistable attractor framing — RETIRED** (the system is not bistable at any gain; §17/§18)
> - **the two-mark chromatin replication — SUPERSEDED**: recomputed on fold-change tracks, H3K27ac
>   holds but **ATAC does not replicate** (§25)
> - **off_target_risk = 0.00 — RETRACTED and resolved**: all four guides fail the specificity gate
>   under the exact Doench-2016 matrix (§24)
> - **leave-cell-line-out — DEMOTED** to an internal diagnostic (leaked whole-panel statistics)
>
> Read the README status table or REVIEW_RESPONSE.md for the current position on any claim below.

This document records **everything**: every data source and the exact endpoint used, every method
in full, every experiment including the ones that failed, every negative result, and every defect
found in the pipeline. Summary of outcomes: [FINDINGS.md]({{ '/reports/findings/' | relative_url }}). Result tables:
[RESULTS.md]({{ '/reports/results-ledger/' | relative_url }}).

Nothing here is synthetic. Where evidence was absent, the analysis abstained and the absence is
recorded rather than filled.

---

## Table of contents

1. [Principles and constraints](#1-principles-and-constraints)
2. [Data acquisition — every source](#2-data-acquisition--every-source)
3. [Modules I–VII](#3-modules-iviii)
4. [Module VIII — Regulatory Attractor Control](#4-module-viii--regulatory-attractor-control-rac)
5. [Development history — three prototypes, two killed](#5-development-history--three-prototypes-two-killed)
6. [Validation design](#6-validation-design)
7. [Long-range chromatin model](#7-long-range-chromatin-model)
8. [The PDAC-minus-healthy disease residual](#8-the-pdac-minus-healthy-disease-residual)
9. [Stage 2 — progression state residual](#9-stage-2--progression-state-residual)
10. [Auxiliary evidence layers](#10-auxiliary-evidence-layers)
11. [Every negative result](#11-every-negative-result)
12. [Every defect found](#12-every-defect-found)
13. [The pre-access seal](#13-the-pre-access-seal)
14. [Complete result index](#14-complete-result-index)
15. [Reproduction](#15-reproduction)

---

## 1. Principles and constraints

| Principle | Enforcement |
|---|---|
| No synthetic data | Every model trains on real data; absent evidence → abstain (certified-negative) |
| No external pretrained model supplies candidate features | Enformer/Borzoi permitted only as hash-locked evaluation baselines |
| Honest provenance | REAL / POINTER / GATED governance; sha256 never fabricated; `pdac verify-data` re-hashes |
| Fail closed | `pdac predeploy` runs 11 gates and exits non-zero on any failure |
| Negatives are results | Failed hypotheses stay in the record |

**Compute.** Single machine, NVIDIA RTX 5070 Ti Laptop GPU (12 GB), torch 2.11.0+cu128,
Python 3.12. The GPU was shared with unrelated workloads throughout (~11.4 GB resident), which
materially slowed and in two cases killed long runs; this is noted where it affected results.

---

## 2. Data acquisition — every source

All corpora are sha256-verified; **14 manifests honest** under `pdac verify-data`. Raw bytes are
never published — only manifests and TrackSpecs.

### 2.1 Pre-existing on disk

| Corpus | Content |
|---|---|
| `tcga-paad` | RSEM expression (177 tumours), somatic mutations (cBioPortal) |
| `gtex-pancreas` | GTEx v8 median TPM |
| `encode-bulk` (95 GB) | 54 healthy-pancreas ChIP BAMs — CTCF ×8, H3K27me3 ×8, H3K36me3 ×8, H3K9me3 ×8, H3K27ac ×7, H3K4me3 ×6, H3K4me1 ×6, H3K9ac ×3 — plus 276 bigWigs |
| `encode-pancreas-atac` / `-h3k27ac` | narrowPeak calls |
| `fantom5-cage`, `gencode-v46`, `lambert-tf` (1,639 TFs), `intogen-pdac`, `doench-2016`, `dbsnp-common`, `hg38/hg19/mm10/mm9` | — |

### 2.2 Acquired during this work

| Layer | Source + exact route | Content |
|---|---|---|
| **CRISPR dependency** | DepMap Chronos (reused on disk) + Model.csv via figshare | 1,208 lines × 18,531 genes; **48 PDAC** lines with CRISPR |
| **Cell-line expression** | DepMap `OmicsExpression.csv` (on disk) | 1,684 lines × 19,205 genes; **54 PDAC** |
| **Motifs** | JASPAR 2024 CORE vertebrates non-redundant PFMs | 879 PFMs → **754** gene-mapped |
| **Copy number** | cBioPortal `paad_tcga_pan_can_atlas_2018_gistic`, `/molecular-profiles/{id}/molecular-data/fetch` | 1,707 genes × **183** tumours |
| **RPPA protein** | cBioPortal `..._rppa`, same endpoint | 122 tumours, 20 TFs |
| **DNA methylation** | cBioPortal `..._methylation_hm450`; `/generic-assay-meta/{id}` → **396,065 CpG probes**; `/generic_assay_data/{id}/fetch` | 15,655 genes' promoter probes, **183** tumours |
| **Proteomics** | CPTAC PDAC (Cao 2021) via `cptac` package, `Pdac().get_proteomics("umich")` | 238 tumours × **11,419** proteins; **651 TFs** |
| **Hi-C 3D** | 4DN `4DNESCCP4KTY` (Hi-C on PANC-1) — released **derived tracks** via each file's `open_data_url` | A/B eigenvector (250 kb), diamond insulation (10 kb), TAD boundaries; hg38 |
| **Single-cell** | TISCH2 `PAAD_CRA001160` (Peng 2019) `_expression.h5` + `_CellMetainfo_table.tsv` | 57,443 cells; **11,401 malignant**, 16,280 stromal, 11,654 immune; 35 patients |
| **PDAC chromatin** | ENCODE **Panc1** (47 released experiments), peaks + signal p-value bigWigs, GRCh38 | ATAC 99,761 · CTCF 58,083 · H3K4me1 54,666 · H3K27me3 46,565 · H3K27ac 45,824 · H3K36me3 35,541 · H3K4me3 23,189 · TCF7L2 10,794 · H3K9me3 3,651 |

### 2.3 Access notes that cost real time

- **ENCODE blocks a `Mozilla/5.0` User-Agent → HTTP 405.** The project's own
  `data.fetch._download` (UA `pdac-circuit-fetch/0.1`) works.
- **ENCODE biosample term is `Panc1`, not `PANC-1`** (the latter 404s).
- `signal.metadata._encode_obj` appends `?format=json`, so it 404s on any `/search/?…` URL that
  already carries a query string — search URLs must be built directly.
- **4DN `@@download` hrefs return HTTP 403.** The open S3 path is in the `open_data_url` field of
  `/files-processed/{acc}/?format=json`.
- **cBioPortal generic-assay endpoints**: the working paths are `/generic-assay-meta/{profileId}`
  and `/generic_assay_data/{profileId}/fetch`. Earlier guesses (`/generic-assay-data/…`) 404 —
  this was originally misdiagnosed as "methylation unavailable, use GDC"; it was a wrong path.
- **pyBigWig does not build on Windows**; `pybigtools` works and the project's `_open_bigwig`
  already supports both backends.
- **Healthy histone bigWigs carry no mark in the manifest note** — targets must be resolved via
  `signal.metadata._encode_file` (cached to `data/raw/encode-bulk/histone_signal_target_map.json`;
  yields H3K4me3 ×11, H3K27ac ×10, and 8 each of H3K27me3/H3K4me1/H3K36me3/H3K9me3, H3K9ac ×2).

### 2.4 Verified sanity of the acquired data

DepMap gene effect is biologically coherent: `KRAS −2.14`, `MYC −2.46`, `KLF5 −0.74` are
dependencies; tumour suppressors `TP53 +0.24`, `SMAD4 +0.02` correctly are not.
Copy number is textbook PDAC: `MYC` 40 % amplified, `GATA6` 30 %, `KRAS` 25 %;
`SMAD4` 68 % deleted, `CDKN2A` 63 %, `TP53` 49 %.

---

## 3. Modules I–VII

Four models trained from scratch on real held-out data:

| model | module | architecture | metric |
|---|---|---|---|
| promoter | II | RF + CNN | Spearman 0.517 |
| enhancer | II | CNN | AUROC 0.815 |
| gRNA on-target | V | GBT + CNN | Spearman 0.494 |
| promoter GAN | VII | WGAN-GP | 4-mer JS 0.009 |

**Module I (target prioritization).** MCDA over [expression, specificity, oncogenic, subtype,
breadth]; weights grid-searched on the simplex to maximise known-driver recovery@k; ranking
significance from a permutation null on the mean rank of controls (B = 2,000).
Result: **top-quartile driver recovery 5/7, permutation p = 0.003**, 177 tumours, 1,321 candidate
TFs, `powered: true`.

**Deep design.** Every distinct circuit is *individually* ODE-simulated rather than sampled from a
combinatorial space. Each circuit's Hill-ODE betas derive from its own parts (TF expression →
`beta_TF`; promoter strength × enhancer activity → `beta_synprom`; guide efficiency → `beta_rep`);
robustness is a real per-circuit parameter sweep and knockdown a real steady-state readout.
Run: **3,003 classical** (1,233 single-TF + 1,770 multi-TF AND-logic; **215** on Pareto front 0)
and **3,003 basal** (**143** on front) = **6,006 circuits**.

**Rigor calibration (by construction).** BH-FDR under partial null 0.041; split-conformal 90 %
coverage 0.897; permutation type-I 0.060; certified-negative lattice consistent.

---

## 4. Module VIII — Regulatory Attractor Control (RAC)

### 4.1 Motivation

A graph neural network that *scores* circuits learns a correlation, not a mechanism. RAC instead
learns the **dynamics** of the regulatory network and treats a synthetic circuit as a **control
input** to those dynamics. Circuit design becomes optimal control of a data-calibrated attractor
landscape.

### 4.2 Formulation

Node activations `x ∈ [0,1]^n` over PDAC transcription factors evolve under a bistable map

```
x  ←  σ( gain · ( W x + b ) ),        gain = 4.0
```

with `W = A ⊙ M` masked to the regulatory graph `M` and sign-anchored to co-expression.
Fixed points satisfy `x* = σ(gain·(W x* + b))`. Strong positive feedback makes the system
**bistable**: a viable high-activation state and a dead low-activation state.

**This bistability is the whole point.** Cell death is collapse to the null attractor, so
*essentiality becomes a dynamical property*: a node is essential if clamping it down collapses the
viable state. A globally contractive system (unique fixed point) cannot express this — an earlier
formulation had exactly that flaw (§5).

**Fitting objective**

```
L = mean( (σ(gain·(S Wᵀ + b)) − S)² )            # observed states are fixed points
  + 0.5 · mean( relu( σ(gain·(x_dead Wᵀ + b)) − 0.25 ) )   # dead state stays low (bistability)
  + 1e-3 · mean( (A ⊙ M)² )                       # L2
```

where `S` is the DepMap PDAC cell-state panel (54 lines), min–max scaled per gene to `[0.02, 0.98]`.
Adam, lr 0.03, 1,800–2,400 epochs.

### 4.3 Graph construction (`attractor/graph.py`)

- **Nodes** — expressed TFs (Lambert catalogue) ∪ subtype-signature ∪ IntOGen drivers ∪ controls,
  filtered to >50 % of PDAC lines expressed, ranked by variance across the PDAC panel.
- **Edges** — **pan-cancer** co-expression across all 1,684 DepMap lines (robust estimation),
  thresholded at |r| > τ, then instantiated on the 54-line PDAC state. Decoupling *graph
  estimation* (needs many samples) from *state* (must be PDAC) is deliberate.
- **Motif refinement** — JASPAR PWMs (log-odds, 0.8 pseudocount, uniform background) scanned
  against each target's hg38 promoter (TSS −2,000/+500) on both strands via a vectorised
  sliding-window einsum; score normalised by the per-position bit-max. Provides **directionality**.
- **Chromatin context** — ENCODE pancreas ATAC/H3K27ac overlap at TSS ± 2 kb.
- **Copy number** — TCGA GISTIC amplification frequency and mean per node.
- **Methylation** — HM450 promoter beta per node (silencing filter).
- **Healthy direction** — sign of GTEx-normal minus TCGA-tumour log2FC: the direction each TF must
  move to become less PDAC-like. Only the *sign* is used, since the comparison is cross-platform.

### 4.4 Collapse essentiality

For each node `i`, clamp `x_i = 0.02` and re-settle from the attractor; collapse is the
network-wide positive drop `Σ_{j≠i} relu(x*_j − x^{KO}_j)`, averaged over per-line attractors.
All `n` knockdowns are evaluated as one batched `(n × n)` settle on GPU.

### 4.5 Control design

Greedy minimal repressible set. CRISPRi can only repress, so candidates are restricted to nodes
with `healthy_dir < 0` (up in tumour) and **not** DepMap-essential. Each step maximises

```
net = Σ relu( (x^ctrl − x*) ⊙ healthy_dir )  −  2 · Σ ( relu(x* − x^ctrl) ⊙ essential_mask )
```

i.e. movement toward the healthy direction penalised by collapse of essential activity.

### 4.6 Motif is annotation, not a fit weight

Up-weighting the fit initialisation by motif support **lowered** the essentiality AUC
(0.63 vs 0.67; CIs overlap heavily, so this is "no benefit", not "harm"). Conflating
promoter-proximity with dynamical importance is not justified, so motif is used for edge
direction and target scoring while **co-expression alone drives the dynamics**
(`motif_weight = 0.0` in the fit).

---

## 5. Development history — three prototypes, two killed

Recorded because the failures shaped the method.

### Prototype 1 — contractive dynamics on TCGA bulk → **killed**

Signed graph map `dx/dt = −x + tanh(Wx + b)` fit to TCGA basal/classical centroids, spectral norm
penalised toward contraction; "load-bearing" defined as network reconfiguration magnitude under
knockdown.

**Result: null.** Collapse-vs-essentiality ρ = **−0.097** (p = 0.12) — *worse* than a degree
baseline (−0.109) and the wrong sign. Top "load-bearing" nodes were STAT4, ZNF831, MAFB, IKZF1 —
**immune** TFs with ~zero essentiality.

**Two diagnoses, both load-bearing for the final design:**
1. **Composition confound** — bulk co-expression across tumours is driven by immune/stromal
   infiltration, so the graph was partly a *composition* graph. DepMap cell lines are pure tumour
   cells; the substrate was changed to cell lines.
2. **Wrong essentiality semantics** — "removal changes the transcriptome a lot" ≠ "cell dies".
   Essentiality was redefined as **collapse to the dead attractor**, which required abandoning
   global contraction in favour of bistability.

### Prototype 2 — bistable dynamics on DepMap PDAC lines → **kept, sign correct**

ρ = **+0.111** (p = 0.059), degree baseline null (−0.013). Directionally right, borderline.
Top collapse nodes became biologically sensible (FOXM1, TEAD4, CEBPB, KMT2D).

### Prototype 3 — pan-cancer graph + PDAC attractor → **adopted**

Graph estimated across all 1,684 lines, state on the 54 PDAC lines. AUC 0.654 vs degree 0.560. **[RETRACTED — §1/§15: the head-to-head test gives dAUC -0.082 and partial rho 0.028 (p=0.56); collapse adds nothing beyond degree.]**
A threshold sweep then showed the signal is robust where powered (§6.2). This is the shipped
design.

**A near-miss worth recording.** Prototype 3 initially reported `permutation p = 0.0003`. That was
an **artifact**: one gene had NaN essentiality, `spearmanr` returned NaN, and `null >= NaN` is
always False, so the permutation count collapsed to `1/3001`. Caught, fixed, and the honest
statistics rebuilt (bootstrap CI + AUC + clean permutation).

---

## 6. Validation design

DepMap CRISPR is **never used to fit the dynamics**. It enters only as a held-out, *cross-modality*
target (dynamics fit to **expression**; asked to predict **loss-of-function**) and as a safety
filter in control design.

### 6.1 Definitive campaign (`scripts/heavy_rac_campaign.py`)

- **Grid**: nodes {400, 600, 800} × co-expression threshold {0.30, 0.35, 0.40}, each an 8-member
  bootstrap ensemble.
- **Definitive run** at the best config: 40-member bootstrap ensemble (resampling cell lines),
  2,400 epochs, **50,000-permutation null**, 5,000-sample bootstrap CI, degree and eigenvector
  baselines.
- **Leave-cell-line-out CV**: 54 refits.
- Incremental JSON writes so partial progress survives interruption.

**Grid finding — sparser, focused graphs win.** Best = 400 nodes / τ = 0.4 (→ 422 nodes,
5,050 edges). *Adding* lower-variance TFs **lowers** AUC to 0.51–0.56: the collapse signal dilutes.

**Definitive result**

| statistic | value |
|---|---|
| point AUC (essential threshold 0.4) | **0.653** [0.539, 0.759] |
| **50,000-permutation p** | **0.0022** |
| degree / eigenvector baseline | 0.629 / 0.584 |
| 40-member bootstrap ensemble | 0.606 [0.475, 0.663] |

**Honest reading.** Significant at the point estimate; **modest and noisy under heavy resampling**
(not every bootstrap member beats chance); only moderately above the degree baseline. Both numbers
are reported.

### 6.2 Threshold sensitivity (why the AUC is trusted at all)

| threshold | n positive | AUC collapse [95 % CI] | degree | eigenvector | CI excludes chance |
|---|---|---|---|---|---|
| 0.3 | 34 | **0.661** [0.552, 0.765] | 0.621 | 0.584 | **yes** |
| 0.4 | 28 | **0.657** [0.533, 0.773] | 0.624 | 0.584 | **yes** |
| 0.5 | 16 | 0.569 [0.382, 0.750] | 0.533 | — | no (underpowered) |
| 0.6 | 9 | 0.507 [0.278, 0.737] | 0.486 | — | no |

Collapse beats both centrality baselines at **every** threshold. The wide CI at 0.5–0.6 is
underpowering (few positives), not a contrary result.

### 6.3 Leave-cell-line-out — the strongest evidence

Refit on 53 lines; test whether the held-out **real** state is a lower-residual fixed point than a
permuted null of the same state.

| | residual |
|---|---|
| held-out real cell state | **0.064** |
| permuted null | 0.128 |
| Wilcoxon (held-out lower) | **p ≈ 0** |

The dynamics **generalise to PDAC cell lines they were never fit on** — evidence the attractor
structure is real rather than memorised. This, not the essentiality AUC, is the result that
licenses the method.

---

## 7. Long-range chromatin model

### 7.1 Architecture (PDACircuitFormer)

196,608 bp input → 1,536 × 128 bp bins; ~1.95 M trainable parameters. Convolutional stem with
strided downsampling, gated dilated depthwise blocks, **landmark attention** for linear-memory
long-range mixing, continuous assay/state/perturbation conditioning (12/18/22 coordinates), a
factorised state/intervention circuit bottleneck (32 circuit factors), and profile + uncertainty
heads. Verified forward pass: output `[1, 1536]`, finite, peak 0.03 GB CUDA.

### 7.2 Stage 1 — healthy prior (first real training)

Trained from scratch on **all 33,156 compiled healthy-pancreas shards**, bf16 autocast, resumable,
checkpointing.

| | value |
|---|---|
| held-out profile loss | 0.0088 → **0.0079** (~step 700, then plateau) |
| held-out profile **correlation** | **0.726** (single run) |

**Throughput note.** The trainer is **silent on stdout** — progress is only visible via the
checkpoint directory (`scripts/extract_train_curve.py` reconstructs the curve from `step-*.pt`).
`gradient_accumulation` = 1 was required to obtain many optimizer steps: the healthy window set is
a single ~2,600-window pass, so at grad-accum 8 the run terminated after 50 optimizer steps.

**Convergence is a finding, not a failure.** A 1.95 M-parameter model on ~2,600 windows converges
in ~15 min. Running it for hours would re-see the same windows and overfit, not learn. Compute was
therefore spent on **uncertainty quantification** instead.

### 7.3 Multi-seed ensemble (`scripts/chromatin_seed_ensemble.py`)

Independent random-seed retrainings, resumable, each recording the held-out validation.

| n seeds | profile correlation | std | min | max |
|---|---|---|---|---|
| **8** | **0.7102** | **0.0091** | 0.698 | 0.728 |

The model predicts unseen healthy-pancreas chromatin at **r ≈ 0.71**, reproducibly. The estimate
converged: seeds 7 → 8 moved the mean by 0.0007. A 24-seed target was truncated at 8 (§12.4).

---

## 8. The PDAC-minus-healthy disease residual

Until PDAC chromatin was acquired, the corpus was **healthy-only** and the residual the entire
design rests on was unmeasurable.

### 8.1 Why peak overlap cannot measure it

Naive peak overlap gives `pdac_specific_open = 0.00` — an **artifact**: the healthy ATAC peak set
has **874,795** peaks versus PDAC's **99,761** (~9×, a permissive merged set), so "open in PDAC,
closed in healthy" is empty *by construction*. Reported as an artifact, not biology.

A *within-PDAC* enrichment is unaffected by that asymmetry and is valid:

| Fisher, one-sided | targets | background | OR | p |
|---|---|---|---|---|
| accessible in PDAC (ATAC) | 94 % | 73 % | 6.21 | **0.029** |
| active in PDAC (H3K27ac) | 89 % | 63 % | 4.65 | **0.017** |

### 8.2 Signal-level residual (`scripts/pdac_disease_residual.py`)

Matched assays — ENCODE **signal p-value** tracks, Panc1 vs healthy pancreas, same output type and
assembly — `log2((pdac + 0.1)/(healthy + 0.1))` over TSS ± 2 kb, ~1,676 loci, 6 healthy tracks per
mark. Parameterised by `RESIDUAL_MARK`.

| mark | RAC targets | background | all loci | Mann-Whitney p |
|---|---|---|---|---|
| **ATAC** | **+0.279** (70 % up) | −0.263 (43 %) | −0.256 | **0.010** |
| **H3K27ac** | **+1.596** (80 % up) | +0.261 (54 %) | +0.277 | **0.00062** |

~~**Replicated on two independent marks.** The ATAC result is meaningful precisely because the
overall trend runs the **other way** — TF promoters are on average *less* accessible in Panc1 than
healthy pancreas — yet the targets buck it. On H3K27ac the gain is ≈ **3× active-enhancer signal**.~~
The targets were derived with **no chromatin input at all** (that part stands).

**[SUPERSEDED — §25/§28.** These figures come from ENCODE **signal p-value** tracks, which conflate sequencing depth with enrichment. Recomputed on **fold-change over control**: **H3K27ac holds** (+0.919 vs -0.091, p = 0.0022, surviving expression-, selection-variable- and degree-matched backgrounds and 12/12 window x pseudocount settings) but **ATAC does NOT replicate** (p = 0.074). The effect is **promoter-local** — absolute enrichment falls to 1.01x at +/-25 kb — and **modest**, ~1.5-1.8x, not the "3x active-enhancer signal" stated here. The two-mark replication claim is withdrawn. See [docs/ADDENDUM_CHROMATIN.md]({{ '/addenda/chromatin/' | relative_url }}).]**

`sealed_studies_touched: false` is recorded in both artifacts.

---

## 9. Stage 2 — progression state residual

### 9.1 Encoding a PDAC cell line honestly

The frozen 18-coordinate state schema is
`[healthy_pancreas, PanIN, primary_PDAC, metastatic_PDAC, classical, basal_like, treated,
drug_resistant, organoid, PDX, primary_tumor, epithelial_fraction, fibroblast_fraction,
immune_fraction, KRAS_activity, purity_or_confidence, species_human, species_mouse]`.

There is **no `cell_line` coordinate** — but the schema separates *disease state* from *sample
format*, so a cultured line is representable without inventing one or amending the registry:

| coordinate | Panc1 | reason |
|---|---|---|
| `primary_PDAC` | **1** | Panc1 is a ductal adenocarcinoma line (disease state) |
| `primary_tumor` | **0** | cultured cells, **not** tumour tissue |
| epithelial / fibroblast / immune | **1 / 0 / 0** | a pure line has no stroma or immune compartment |
| `KRAS_activity` | **1** | Panc1 carries KRAS G12D |
| `purity_or_confidence`, `species_human` | 1, 1 | — |

Mislabelling the line as primary tumour would corrupt the residual. Assay descriptors are produced
by the project's **own `encode.assay_vector`**, never hand-rolled, keeping the 12 frozen assay
coordinates identical to the healthy corpus. (Note: `assay_vector` has no slot for **H3K36me3**,
which therefore cannot be compiled.)

### 9.2 The pairing requirement

`paired_residual_loss(residual, paired_delta, pair_mask)` needs `paired_delta`, which only exists
for **paired** shards. `compose_paired_shards` requires a **non-empty, identical `pair_group`** on
both sides and exactly `state_reference` (normal) / `state_treatment` (disease) relations.

Pipeline: `make_panc1_track_specs.py` → `build_pdac_paired_specs.py` → compile **both** sides over
identical coordinates → `resort_shards_for_pairing.py` (§12.2) → `chromatin-pair --mode state`
(40 paired shards, `valid: true`) → train `--stage progression_state_residual --initialize-from`
the healthy prior.

### 9.3 Result

| supervision | unpaired | **paired** |
|---|---|---|
| `residual_delta` (PDAC-minus-normal) | **0.000000** — silently inert | **+0.445470 — ACTIVE** |
| profile / correlation | 0.012 / 0.70 | 0.061 / 0.80 |

**Scope.** The paired run reached **250 steps** before a native crash (exit 127) under GPU
contention. This is **proof the objective is live and correctly wired — not a converged Stage-2
model**. Panc1 is a cell line, so this is a PDAC-*line* residual, not primary tumour.

---

## 10. Auxiliary evidence layers

### 10.1 DNA methylation — silencing filter

HM450 promoter probes only (`TSS200`, `TSS1500`, `1stExon`, `5'UTR`), beta averaged per gene.
Of 413 covered RAC nodes, **72 are hypermethylated** (β > 0.5; median β 0.085).

| gene | β | reading |
|---|---|---|
| **TP63** | **0.733** | the **basal** master TF — silenced, consistent with a classical-dominant cohort |
| **AGR2** | **0.586** | classical marker, hypermethylated → **demoted** despite 28 % amplification |
| GATA6 / KLF5 / MYC / E2F1 / SETDB1 | 0.02–0.05 | unmethylated → genuinely active |
| **CDKN2A** | **0.038** | **not** methylated — silenced by **deletion** (63 %) instead |

The last row is the satisfying one: two different silencing mechanisms cleanly resolved by two
independent layers. Applied to the top-20 targets: 2 hypermethylated (AGR2, FAM83A) flagged as
poor CRISPRi targets; **0 ATAC-open-but-silenced** false positives.

A **mapping bug was found and fixed**: HM450 `NAME`/`DESCRIPTION` are *positionally paired*
(`"A;B"` ↔ `"Body;TSS200"`). Assigning a probe to every listed gene when *any* region is
promoter-like fabricates hypermethylation — KRT19 read β = 0.671, implausible for a highly
expressed ductal marker. After strict positional pairing, KRT19/VIM correctly drop to
no-promoter-coverage and the real signals (TP63, AGR2, SETDB1) are unchanged.

### 10.2 CPTAC proteomics — protein-level TF activity

**651 of 1,639 Lambert TFs quantified at protein level — 32× the 20-TF RPPA panel.** Detection
rate is reported alongside abundance, because a TF that is mRNA-high but protein-undetected is a
poor CRISPRi target.

| gene | protein mean | detected in | reading |
|---|---|---|---|
| **AGR2** | **−0.489** | **100 %** | reliably measured and **low** — converges with its hypermethylated promoter |
| SETDB1 / SF3B1 / SMAD3 / ATM | ~0.0 to +0.07 | 100 % | robustly present |
| GATA6 | +0.057 | 67 % | present |
| **FOSL1 / ZNF331** | — | **15 % / 16 %** | barely detectable → deprioritised |

### 10.3 Hi-C 3D layer

A/B compartment eigenvector, diamond insulation and TAD-boundary distance at each node's TSS,
computed for **1,676 genes** from the released derived tracks (the 2.6 GB contact matrix was not
needed).

**Result: a sanity check that passes, not a discriminator.** 90 % of RAC targets sit in the active
**A** compartment vs 70 % of background TFs — but **Mann-Whitney p = 0.504**, because expressed TFs
are already mostly in A. No claim made.

### 10.4 Single-cell in-vivo malignant substrate

Malignant cells only (11,401), pseudobulked per patient (24 patients with ≥ 30 malignant cells) →
an *in-vivo* tumour-intrinsic graph with no stroma, no immune, no culture.

| substrate | AUC core-essential |
|---|---|
| cultured DepMap cell lines | **0.65–0.66** |
| in-vivo malignant cells | **0.51–0.55** |

**The cultured cell-line graph beats the in-vivo tumour graph** — the substrate that matches the
cell-line-measured CRISPR readout wins. In-vivo is not automatically better.

---

## 11. Every negative result

| # | Hypothesis | Outcome |
|---|---|---|
| 1 | Contractive dynamics on TCGA bulk predict essentiality | **Null** (ρ = −0.097, worse than degree). Composition confound + wrong essentiality semantics |
| 2 | Model regulatory coupling predicts DepMap **co-essentiality** | **Null** — 39,340 gene pairs, ρ ≈ −0.01; top-1 % coupled pairs mean co-essentiality −0.003 vs +0.003 background |
| 3 | RAC predicts PDAC-**selective** dependency | **Null** (AUC ≈ 0.49) |
| 4 | An in-vivo malignant graph rescues the selective signal | **Noise** — apparent AUC 0.63–0.66 came from **3 positives** and collapsed to 0.43 at another threshold |
| 5 | The **composition confound** caps the selective signal | **Refuted by our own data** — power audit: of **1,164 TFs** with DepMap CRISPR, only **7** are PDAC-selective at sel > 0.25 (5 at > 0.30; 30 even at a lenient > 0.10), vs 42 core-essential at abs > 0.40. It is a **biological/power ceiling**; no graph substrate can resolve it with this readout |
| 6 | Motif up-weighting improves the dynamics fit | **No benefit** (AUC 0.63 vs 0.67, CIs overlap) → motif demoted to annotation |
| 7 | Hi-C A/B compartment discriminates targets | **Not a discriminator** (p = 0.50) |
| 8 | Peak overlap can measure the PDAC-specific differential | **Artifact** (0.00 by construction, 9× peak-set asymmetry) |

Negative #5 is the most consequential: it overturned a hypothesis stated earlier in this work and
redirects the programme away from a question that cannot be answered with this readout.

---

## 12. Every defect found

### 12.1 Unpaired data makes Stage 2 a silent no-op

`residual_delta` reads exactly **0** at every step while the profile loss still falls — a run can
**look successful while learning no residual at all**. Only inspecting the loss *parts* reveals it.

### 12.2 `chromatin-compile` and `chromatin-pair` disagree on chromosome order

Compile emits **lexicographic** order (chr1, chr10 … chr19, **chr2**, chr20 …);
`pairing._coordinate_key` sorts **numerically** (`chr2` → (0,2) precedes `chr19` → (0,19)). The
stream therefore runs *backwards* at chr19 → chr2 and pairing aborts with "shards are not
coordinate-sorted". **The paired-state path could not run on ENCODE-compiled human data at all.**
Fixed by `scripts/resort_shards_for_pairing.py`, which reorders rows into `_coordinate_key` order
and changes nothing else; frozen compile behaviour is untouched.

### 12.3 TCGA expression cache overwritten by a gene-superset request

`TCGA_EXPR` **is** the raw file `data/raw/tcga-paad/tcga_paad_rsem.csv`. Calling
`load_tcga_paad_expression(genes)` with genes beyond the cache triggers a cBioPortal re-fetch that
**overwrites the frozen raw file**, breaking its sha256 honesty gate. Fixed: RAC now reads the
cache **read-only** (`load_tcga_paad_expression()` with no args); the manifest was re-recorded with
a dated provenance note. Module I metrics were unaffected (recovery 5/7, p = 0.003 unchanged).

### 12.4 The environment kills long background jobs

Long-running jobs are terminated (~15 min – 1.5 h, exit 127), and a `nohup … &` launched *inside* a
tool call is orphaned and dies. Consequences: the 24-seed ensemble truncated at 8; the paired
Stage-2 run died at 250 steps. Mitigation: long scripts are **resumable** and write incrementally.
Additionally the GPU was shared with unrelated workloads (~11.4 GB, 100 % util), which turned a
2-minute RAC run into 40 + minutes.

### 12.5 Smaller ones

- Permutation p poisoned by a NaN (§5) — `null >= NaN` is always False, yielding a spurious
  `p = 1/3001`.
- JASPAR headers are `>MATRIX_ID NAME`; keying motifs by token 0 gives matrix IDs, not gene
  symbols (879 → 0 usable until fixed).
- The TrackSpec loader **rejects unknown fields** (a `sample_format` annotation was refused — the
  honesty contract working as designed).
- `chromatin-pair` requires a `manifest.json` beside the shards and refuses a pre-existing output
  directory.
- A Windows file-lock race on `best.pt.tmp → best.pt` kills a training seed; retried.

---

## 13. The pre-access seal

`results/frozen/protected-studies.seal.json` is `SEALED_BEFORE_ACCESS` and protects **five**
`external_study_test` studies:

| study | role |
|---|---|
| GSE301272 | protected external **KLF5** accessibility perturbation |
| GSE301284 | protected external **KLF5** active-chromatin perturbation |
| GSE295354 | protected independent-lab external **KLF5** lineage perturbation |
| GSE124229 / GSE124230 | primary human external validation |

Release requires freezing the three registered candidate seeds
(`20260620`, `20260714`, `20260808`); `target_download_release_required: true`.

**These were not downloaded.** Three of the five are KLF5 perturbation studies, and KLF5 is the TF
this model ranks first — downloading them (or the lookalike series GSE310807/309861/309860) would
unblind the test set for the very prediction the model most wants credit for. Analyses record
`sealed_studies_touched: false`.

**Hard fact established while looking for an alternative:** GDC carries ATAC-seq for **23 TCGA
projects and PAAD is not among them** (Corces 2018 excluded pancreatic). Primary PDAC chromatin
exists only on GEO.

**Independent corroboration.** Three 2024–25 GEO series are titled *"KLF5 controls
subtype-independent highly interactive enhancers in PDAC"* — the literature independently converged
on the master regulator this unsupervised model ranked first.

---

## 14. Complete result index

| artifact | contents |
|---|---|
| `results/run_{basal,classical}.json` | Modules I–VI, verdict OK / cert real |
| `results/deep_{basal,classical}.json` | 3,003 individually simulated circuits each |
| `results/attractor_{map,validation,control,targets}.json` | RAC graph, validation, control set, target catalogue |
| `results/rac_campaign.json` | grid + 40-fit ensemble + 50k permutation + LOO-CV |
| `results/chromatin_training_curve.json` | healthy-prior loss curve |
| `results/chromatin_ensemble.json` | multi-seed profile correlation |
| `results/pdac_disease_residual_{ATAC-seq,H3K27ac}.json` | the disease residual, both marks |
| `results/pdac_vs_healthy_chromatin.json` | PDAC vs healthy peak-level enrichment |
| `results/hic_3d_layer.json` | A/B compartment, insulation, TAD distance |
| `results/single_cell_malignant_graph.json` | in-vivo malignant substrate + honest negative |
| `results/chromatin_inventory_current.json` | corpus audit |
| `figures/fig_{models,gan,pareto,deep,chromatin,attractor,chromatin_training}.png` | — |

---

## 15. Reproduction

```bash
pdac fetch-data --all-open
pdac verify-data                                   # 14 manifests, sha256
pdac train --all
pdac run-pipeline --subtype {basal,classical}
pdac run-deep --subtype classical --multi-top 60
pdac attractor-design                              # Module VIII
pdac chromatin-model-info --forward-check --device cuda
pdac figures
pdac predeploy                                     # 11 fail-closed gates

python scripts/heavy_rac_campaign.py               # grid + ensemble + 50k perm + LOO-CV
RESIDUAL_MARK=H3K27ac python scripts/pdac_disease_residual.py
python scripts/hic_3d_layer.py
python scripts/build_malignant_graph.py
python scripts/make_panc1_track_specs.py           # PDAC TrackSpecs (frozen schema)
python scripts/build_pdac_paired_specs.py          # paired reference/treatment specs
python scripts/resort_shards_for_pairing.py <glob> <out>
```

Long-range training (Stage 1 → Stage 2):

```bash
pdac chromatin-train --stage healthy_prior \
  --shards "data/processed/chromatin_encode_healthy_full_v1/*/*.npz" \
  --fasta data/raw/hg38-ref/hg38.fa --device cuda

pdac chromatin-pair --mode state \
  --reference "<sorted reference>/*.npz" --treatment "<sorted treatment>/*.npz" \
  --output data/processed/chromatin_pdac_paired_v1

pdac chromatin-train --stage progression_state_residual \
  --shards "data/processed/chromatin_pdac_paired_v1/*.npz" \
  --initialize-from models/chromatin/healthy_prior_ckpt/best.pt \
  --fasta data/raw/hg38-ref/hg38.fa --device cuda
```

The trainer is stdout-silent; use `scripts/extract_train_curve.py <ckpt_dir> <out.json>` to
reconstruct the loss curve from checkpoints.

---

## 16. What is new, stated precisely

Circuit design is treated as **optimal control of a data-calibrated regulatory attractor
landscape**, where essentiality is a *dynamical* property (collapse to the dead attractor) rather
than a centrality statistic, and the dynamics are required to agree with independent evidence
streams — bulk/cell-line steady states as attractors, CRISPR loss-of-function as held-out truth,
and chromatin as the counterfactual substrate.

The claim is licensed by **leave-cell-line-out generalisation** (§6.3) and the **two-mark disease
residual** (§8.2) — not by the essentiality AUC alone, which is honestly modest (0.65, fragile
under bootstrap, only moderately above a degree baseline).

The pipeline **does not beat Enformer** and the benchmark **ABSTAINs** by construction.
