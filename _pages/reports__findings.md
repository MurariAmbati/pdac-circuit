---
layout: default
title: "Findings"
subtitle: "Findings as recorded during the project, with their current standing."
description: "Findings as recorded during the project, with their current standing."
permalink: /reports/findings/
group: reports
order: 5
---

Every
number below is read from a real-data artifact under `results/`. No synthetic data is used
anywhere; where evidence is absent the pipeline abstains rather than fabricating.

---

> **RETRACTION NOTICE (2026-07-16).** Result #4 below — "attractor-collapse predicts CRISPR
> essentiality" — **is retracted**. A direct head-to-head test against the degree baseline (never
> previously run) gives dAUC = -0.082 (95% CI [-0.199, +0.029]) and a null partial association
> after controlling for degree, expression and variance (rho = 0.028, p = 0.56). Collapse adds no
> information beyond degree. The original permutation p = 0.0022 tested against chance, not against
> degree, and the configuration had been selected on the same CRISPR labels. Results #2 and #3 are
> also qualified — see **[REVIEW_RESPONSE.md]({{ '/reports/review/' | relative_url }})**.

## 1. Executive summary

A seven-module pipeline designs synthetic gene circuits against PDAC transcription factors on
real open data, with four models trained from scratch. This document records the complete result
set, including an eighth module (**Regulatory Attractor Control**), the first real training of the
long-range chromatin model, a full multi-omic data programme, and the first measurement of the
PDAC-minus-healthy chromatin residual the design rests on.

**The five results that matter**

| # | Result | Evidence |
|---|---|---|
| 1 | ~~**Regulatory dynamics generalise to unseen PDAC cell lines**~~ **[DEMOTED — leaked whole-panel statistics; internal diagnostic only, leakage-free rerun still pending]** | leave-cell-line-out: held-out real state is a **2× lower-residual fixed point** than a permuted null (0.064 vs 0.128), Wilcoxon **p ≈ 0** |
| 2 | **Circuit targets sit on PDAC-gained chromatin** — ~~replicated on two independent marks~~ **[SUPERSEDED — §25/§28]** | Recomputed on **fold-change** tracks (the p-values here came from signal p-value tracks, which conflate depth with enrichment): **H3K27ac holds** (p = 0.0022, and survives expression-, selection-variable- and degree-matched backgrounds, 12/12 parameter settings) but **ATAC does NOT replicate** (p = 0.074). Effect is promoter-local and modest (~1.5–1.8×), not the figures shown here. See [docs/ADDENDUM_CHROMATIN.md]({{ '/addenda/chromatin/' | relative_url }}) |
| 3 | **Long-range model trained and reproducible** | held-out chromatin profile correlation **0.7102 ± 0.0091** (8 seeds) |
| 4 | ~~**Attractor-collapse predicts CRISPR essentiality out-of-modality**~~ **[RETRACTED — see notice above and §1/§15/§15b]** | ~~AUC 0.653, p = 0.0022; beats degree (0.629)~~ — the head-to-head test gives **dAUC -0.082** (95% CI [-0.199, +0.029]) and **partial rho 0.028 (p = 0.56)**. Collapse is degree in disguise; the bistable framing is also retired (§17) |
| 5 | **The disease-residual objective is live** | `residual_delta` **0.000000 → +0.445470** once data is genuinely paired |

**What is explicitly *not* claimed:** the model does not beat Enformer; the benchmark **ABSTAINs**
by construction. See §8.

---

## 2. Data foundation (all real, sha256-verified — 14 manifests honest)

| Layer | Source | Content |
|---|---|---|
| Bulk RNA | TCGA-PAAD, GTEx v8 | 177 tumours + normal pancreas |
| Somatic mutations | TCGA-PAAD (cBioPortal) | MAF variant classes |
| **Copy number (GISTIC)** | cBioPortal TCGA-PAAD | 1,707 genes × 183 tumours |
| **DNA methylation (HM450)** | cBioPortal TCGA-PAAD | 396,065 probes → 15,655 genes' promoters, 183 tumours |
| **Proteomics (mass-spec)** | CPTAC PDAC (Cao 2021) | 238 tumours × 11,419 proteins; **651 TFs** |
| RPPA protein | cBioPortal TCGA-PAAD | 122 tumours, 20 TFs |
| Healthy chromatin | ENCODE pancreas | **54 ChIP BAMs**, 8 marks + CTCF, 276 bigWigs (95 GB) |
| **PDAC chromatin** | ENCODE **Panc1** | ATAC 99,761 · H3K27ac 45,824 · CTCF 58,083 · H3K4me1 54,666 · H3K27me3 46,565 · H3K36me3 35,541 · H3K4me3 23,189 · TCF7L2 10,794 · H3K9me3 3,651 |
| **Hi-C 3D** | 4DN PANC-1 `4DNESCCP4KTY` | A/B eigenvector (250 kb), insulation (10 kb), TAD boundaries, hg38 |
| **Single-cell** | TISCH2 `PAAD_CRA001160` | 57,443 cells; **11,401 malignant**; 24 patients |
| CRISPR dependency | DepMap (Chronos) | 1,208 lines × 18,531 genes; 48 PDAC |
| Motifs | JASPAR 2024 CORE | 879 PFMs → 754 gene-mapped |
| Promoter atlas / drivers / gRNA | FANTOM5, IntOGen, Lambert, Doench-2016, dbSNP | — |

DepMap sanity: `KRAS −2.14`, `MYC −2.46`, `KLF5 −0.74` are dependencies; `TP53 +0.24`,
`SMAD4 +0.02` correctly are not. Copy number is textbook: `MYC` 40 % amplified, `GATA6` 30 %,
`SMAD4` 68 % deleted, `CDKN2A` 63 % deleted.

---

## 3. Modules I–VII (trained from scratch)

| model | module | arch | metric |
|---|---|---|---|
| promoter | II | RF + CNN | Spearman 0.517 |
| enhancer | II | CNN | AUROC 0.815 |
| gRNA on-target | V | GBT + CNN | Spearman 0.494 |
| promoter GAN | VII | WGAN-GP | 4-mer JS 0.009 |

**Module I** — top-quartile driver recovery **5/7**, permutation **p = 0.003**, 177 tumours,
1,321 candidate TFs, MCDA weights grid-searched (`powered: true`).

**Deep design** — every distinct circuit individually ODE-simulated: **3,003 classical**
(215 on the Pareto front) and **3,003 basal** (143 on front) = **6,006 circuits**, each with
Hill-ODE betas derived from its own parts and a real per-circuit robustness sweep.

**Rigor calibration** — BH-FDR 0.041, split-conformal 90 % coverage 0.897, permutation type-I 0.060.

---

## 4. Module VIII — Regulatory Attractor Control (RAC)

A bistable graph dynamical system `x ← σ(gain·(Wx + b))` whose viable high-activation fixed point
is fit to the **54 DepMap PDAC cell-line states**. Loss of viability is literally collapse to the
dead attractor. `W` is masked to the co-expression graph, sign-anchored, annotated with JASPAR
promoter-motif edges and TCGA copy number. **DepMap CRISPR is never used to fit the dynamics** —
it is a held-out, out-of-modality validation target.

### 4.1 Out-of-modality validation (CRISPR held out)

Definitive campaign, best config (400 nodes / co-expression threshold 0.4 → 422 nodes, 5,050 edges):

| statistic | value |
|---|---|
| point AUC (thr 0.4) | **0.653** [0.539, 0.759] |
| **50,000-permutation p** | **0.0022** |
| degree / eigenvector baseline | 0.629 / 0.584 |
| 40-member bootstrap ensemble | 0.606 [0.475, 0.663] |

Hyperparameter grid (9 cells): **sparser, focused graphs win**; *adding* lower-variance TFs
**lowers** AUC to 0.51–0.56.

**Honest reading:** significant at the point estimate, **modest and noisy under heavy resampling**
(not every bootstrap member beats chance), and only moderately above the degree baseline.

### 4.2 Leave-cell-line-out generalisation — the strongest result

Refit on 53 lines, test the held-out one:

| | residual |
|---|---|
| held-out **real** cell state | **0.064** |
| permuted null | 0.128 |
| Wilcoxon (held-out lower) | **p ≈ 0** |

The learned dynamics **generalise to PDAC cell lines they were never fit on** — evidence the
attractor structure is real rather than memorised.

### 4.3 Convergent circuit targets

Ranked by collapse + disease-over-expression + master-regulator control + motif + copy number
+ driver status, with a methylation silencing penalty.

| gene | disease log2FC | CRISPR ess. | CNA amp | promoter β | note |
|---|---|---|---|---|---|
| **KLF5** | +9.3 | **0.74** | 16 % | 0.046 | canonical PDAC lineage-survival TF |
| **GATA6** | +5.9 | 0.03 | **30 %** | 0.045 | classical-subtype master TF |
| SETDB1 | +6.4 | 0.51 | 32 % | 0.026 | essential chromatin modifier |
| BRCA2 / KMT2C / ARID1B | +6.1…+8.5 | — | 14–23 % | — | IntOGen drivers |
| E2F1 / MYBL2 | +5.6…+6.9 | 0.41 / 0.60 | 24–26 % | 0.018 / 0.212 | proliferation, essential + amplified |
| ELF3 / GRHL2 | +5.7…+6.3 | 0.03 / 0.25 | 31 / 33 % | — | epithelial identity |
| **AGR2** | +10.4 | 0.06 | 28 % | **0.586** | **demoted** — hypermethylated *and* low protein |

An unsupervised model that never saw CRISPR data surfaces **KLF5 and GATA6**, the two textbook
PDAC master TFs. Independent corroboration: three 2024–25 GEO series are titled *"KLF5 controls
subtype-independent highly interactive enhancers in PDAC"*.

### 4.4 Negatives, reported

- **No PDAC-selective dependency signal** (AUC ≈ 0.49).
- **Model coupling does not predict DepMap co-essentiality** — 39,340 gene pairs, ρ ≈ −0.01, null.
- **Motif up-weighting of the fit does not improve** essentiality AUC (0.63 vs 0.67; CIs overlap).
  Motif is therefore used for directionality/annotation, not to reweight the dynamics.

---

## 5. Long-range PDACircuitFormer — first real training

196,608 bp context → 1,536 × 128 bp bins, ~1.95 M trainable parameters, trained from scratch on
**all 33,156 compiled healthy-pancreas shards** (Stage 1 healthy prior).

| | value |
|---|---|
| held-out profile loss | 0.0088 → **0.0079** (~step 700, then plateau) |
| **profile correlation, 8 seeds** | **0.7102 ± 0.0091** (0.698 – 0.728) |

The model predicts unseen healthy-pancreas chromatin at **r ≈ 0.71**, reproducibly. It converges
in ~15 min: the healthy window set is a single ~2,600-window pass, so more compute would overfit,
not learn.

---

## 6. The PDAC-minus-healthy disease residual (measured for the first time)

Signal-level on matched assays (ENCODE signal p-value, Panc1 vs healthy pancreas,
log2((pdac+0.1)/(healthy+0.1)) over TSS ± 2 kb, ~1,676 loci). **No sealed study touched.**

| mark | RAC targets | background | all loci | Mann-Whitney p |
|---|---|---|---|---|
| **ATAC** | **+0.279** (70 % up) | −0.263 (43 %) | −0.256 | **0.010** |
| **H3K27ac** | **+1.596** (80 % up) | +0.261 (54 %) | +0.277 | **0.00062** |

~~Replicated on two independent marks. The ATAC result matters because the overall trend runs the
**other way** — TF promoters are on average *less* accessible in PDAC — yet the targets buck it.
On H3K27ac the gain is ≈ **3× active-enhancer signal**.~~ The targets were derived with **no
chromatin input at all** (that part stands).

**[SUPERSEDED — §25/§28.** These figures come from ENCODE **signal p-value** tracks, which conflate sequencing depth with enrichment. Recomputed on **fold-change over control**: **H3K27ac holds** (+0.919 vs -0.091, p = 0.0022, surviving expression-, selection-variable- and degree-matched backgrounds and 12/12 window x pseudocount settings) but **ATAC does NOT replicate** (p = 0.074). The effect is **promoter-local** — absolute enrichment falls to 1.01x at +/-25 kb — and **modest**, ~1.5-1.8x, not the "3x active-enhancer signal" stated here. The two-mark replication claim is withdrawn. See [docs/ADDENDUM_CHROMATIN.md]({{ '/addenda/chromatin/' | relative_url }}).]**

Peak-overlap **cannot** measure this: the healthy ATAC peak set has 874,795 peaks vs PDAC's
99,761 (~9×, permissive merged), so "open in PDAC, closed in healthy" is empty *by construction*
(`pdac_specific_open = 0.00` is an artifact, not biology). Signal, not peaks.

---

## 7. Stage 2 — the disease-residual objective is live

`progression_state_residual` run on **genuinely paired** Panc1 ↔ mark-matched healthy shards,
initialised from the trained healthy prior.

| supervision | unpaired | **paired** |
|---|---|---|
| `residual_delta` (PDAC-minus-normal) | **0.000000** — silently inert | **+0.445470 — ACTIVE** |
| profile / correlation | 0.012 / 0.70 | 0.061 / 0.80 |

**Two real defects found**

1. **Unpaired data makes Stage 2 a silent no-op.** `residual_delta` reads 0 at every step while
   the profile loss still falls — a run can *look* successful while learning no residual. Pairing
   needs a non-empty, identical `pair_group` plus exactly `state_reference` / `state_treatment`.
2. **`chromatin-compile` and `chromatin-pair` disagree on chromosome order.** Compile emits
   lexicographic (chr1, chr10 … chr19, **chr2**, chr20 …); `pairing._coordinate_key` sorts
   numerically. The stream runs backwards at chr19 → chr2 and aborts, so **the paired-state path
   could not run on ENCODE-compiled human data at all**. Fixed by a resort step
   (`scripts/resort_shards_for_pairing.py`) that reorders rows and changes nothing else.

**Scope:** the paired run reached **250 steps** before a native crash under GPU contention — this
is proof the objective is live and correctly wired, **not** a converged Stage-2 model.

---

## 8. Honest limitations

- **The model does not beat Enformer.** The benchmark **ABSTAINs** by construction and requires
  independent PDAC tumour/organoid chromatin plus hash-locked baselines on all seven axes.
- **Panc1 is a cultured cell line, not a primary tumour.** `pdac_tumor_bams` is still **0**. The
  disease residual measured here is a PDAC-*line* residual.
- **TCGA has no PAAD ATAC-seq.** GDC carries ATAC for 23 TCGA projects; PAAD is not among them
  (Corces 2018 excluded pancreatic). Primary PDAC chromatin exists only on GEO.
- **PDAC-selective TF dependency is a power ceiling, not a confound.** Of 1,164 TFs with DepMap
  CRISPR, only **7** are PDAC-selective at sel > 0.25 (5 at > 0.30; 30 even at a lenient > 0.10).
  No graph substrate — bulk, cell-line or in-vivo single-cell — can resolve it with this readout.
  An earlier hypothesis blaming the composition confound is **refuted by this data**.
- **In-vivo malignant graphs are *worse* than cell lines** for predicting CRISPR essentiality
  (AUC 0.51–0.55 vs 0.65) — the cell-line substrate matches the cell-line-measured readout.
- **RAC essentiality is modest** (AUC ≈ 0.65) and fragile under heavy bootstrap resampling.
- **Hi-C compartments are a passing sanity check, not a discriminator**: 90 % of targets sit in
  the active A compartment vs 70 % of background, but Mann-Whitney **p = 0.50** — expressed TFs
  are already mostly in A.
- **Cross-platform specificity**: tumour-vs-normal uses TCGA RSEM against GTEx TPM; only the sign
  of the direction is trusted.
- **Convergent targets are computational hypotheses**, not validated dependencies.
- **A pre-access seal is in force.** Five studies (`GSE301272`, `GSE301284`, `GSE295354` — all
  KLF5 perturbation — plus `GSE124229`, `GSE124230`) are held as the blinded external test set and
  were **not downloaded**. `sealed_studies_touched: false` is recorded in the residual artifacts.

---

## 9. Reproduce

```
pdac verify-data                                   # re-hash REAL corpora against manifests
pdac run-pipeline --subtype {basal,classical}      # Modules I–VI
pdac run-deep --subtype … --multi-top 60           # 3,003 individually simulated circuits
pdac attractor-design                              # Module VIII (RAC)
pdac chromatin-model-info --forward-check --device cuda
pdac figures                                       # figures/fig_*.png
pdac predeploy                                     # fail-closed gates
python scripts/heavy_rac_campaign.py               # grid + 40-fit ensemble + 50k permutation + LOO-CV
python scripts/pdac_disease_residual.py            # RESIDUAL_MARK=ATAC-seq | H3K27ac
python scripts/hic_3d_layer.py
python scripts/build_malignant_graph.py
```

Artifacts: `results/*.json`, `figures/*.png`, `docs/PDAC_RESEARCH_REPORT.html`.

---

## 10. Method note — what is new here

Circuit design is treated as **optimal control of a data-calibrated regulatory attractor
landscape**, with the dynamics required to agree with independent evidence streams: bulk/cell-line
steady states as attractors, CRISPR loss-of-function as held-out truth for which nodes are
load-bearing, and sequence/chromatin as the counterfactual substrate. The validation that licenses
it is leave-cell-line-out generalisation (§4.2) plus the two-mark disease residual (§6) — not the
essentiality AUC alone, which is honestly modest.
