---
layout: default
title: "Results ledger"
subtitle: "The complete result ledger, including entries later retracted or superseded."
description: "The complete result ledger, including entries later retracted or superseded."
permalink: /reports/results-ledger/
group: reports
order: 4
---

All numbers below are read directly from trained-model
manifests and pipeline run artifacts (real open data; models trained from scratch).

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

## Trained models (real held-out data)

| model | module | arch | metric | cert | data lineage |
|---|---|---|---|---|---|
| promoter | II | rf+cnn | Spearman **0.5275** (full FANTOM5 209k, 60k cap removed; fixed chr8/9 test 0.5199→0.5275, CNN carries it 0.499→0.525; near data-saturation ceiling) | real | fantom5-cage, hg38-ref |
| enhancer | II | cnn | AUROC **0.815** (full uncapped pancreas, 20k→80k actives; fixed chr8/9 test 0.809→0.815; generalizes to PANC-1 PDAC at 0.835) | real | encode-pancreas-atac, encode-pancreas-h3k27ac, hg38-ref |
| grna_ontarget | V | gbt+cnn | Spearman **0.657** (was 0.494; +Kim-2019 real HT → 18,142 guides; CNN retrained on Kim 0.39→0.62, GBM 0.65, balanced 0.40/0.60 ensemble) | real | doench-2016 + kim-2019 |
| promoter_gan | VII | wgan-gp | retrained on full 52k top-quartile real promoters (was 12k); certified-real (4-mer JS 0.012 ≪ 0.05, beats random 0.051) with a stronger selectable tail p90 0.94→**0.99**, median uplift −0.02→**+0.11** | real | fantom5-cage, hg38-ref |

All three learned sequence models were data-limited by artificial training caps; removing the caps (or
adding a second real HT dataset) raises each on a fixed held-out test. Full controlled scaling curves,
cross-domain transfer, and the honest PANC-1 negative are in [docs/ADDENDUM_DATA_SCALING.md]({{ '/addenda/data-scaling/' | relative_url }}).

## Rigor calibration (by construction)

- BH-FDR under partial null: **0.041** (<= 0.06)
- Split-conformal 90% coverage: **0.897**
- Permutation type-I (global null): **0.060**
- Certified-negative lattice: equiv→cert-neg = True, different→not = True

## Designed circuits — Classical subtype

- Module I: top-quartile driver recovery 5/7, permutation p=0.0030, n_tumors=177
- verdict **OK**, cert **real**, data_class **REAL**, 8 circuits

| rank | target TF | efficacy | specificity | robustness | safety | guide on-target | off-risk | circuit cert |
|---|---|---|---|---|---|---|---|---|
| 0 | CREB3L1 | 0.95 | 0.90 | 1.00 | 0.90 | 0.97 | 0.00 | real |
| 0 | HNF4G | 0.94 | 0.89 | 1.00 | 0.90 | 0.94 | 0.00 | real |
| 0 | TCF7L2 | 0.87 | 0.87 | 1.00 | 0.90 | 0.86 | 0.00 | real |
| 0 | KLF5 | 0.90 | 0.52 | 1.00 | 0.90 | 0.79 | 0.00 | real |
| 1 | GATA6 | 0.92 | 0.86 | 1.00 | 0.88 | 0.97 | 0.00 | real |
| 1 | KLF3 | 0.91 | 0.82 | 1.00 | 0.89 | 0.89 | 0.00 | real |

## Designed circuits — Basal subtype

- Module I: top-quartile driver recovery 5/7, permutation p=0.0030, n_tumors=177
- verdict **OK**, cert **real**, data_class **REAL**, 8 circuits

| rank | target TF | efficacy | specificity | robustness | safety | guide on-target | off-risk | circuit cert |
|---|---|---|---|---|---|---|---|---|
| 0 | HMGA1 | 0.97 | 0.83 | 1.00 | 0.86 | 0.99 | 0.00 | real |
| 0 | RARG | 0.91 | 0.83 | 1.00 | 0.90 | 0.82 | 0.00 | real |
| 0 | BHLHE40 | 0.95 | 0.79 | 1.00 | 0.89 | 0.97 | 0.00 | real |
| 0 | TGIF1 | 0.94 | 0.48 | 1.00 | 0.90 | 0.88 | 0.00 | real |
| 0 | SMAD3 | 0.94 | 0.47 | 1.00 | 0.90 | 0.91 | 0.00 | real |
| 1 | YBX3 | 0.89 | 0.81 | 1.00 | 0.90 | 0.79 | 0.00 | real |

## Deep design — Classical subtype (individually-simulated circuits)

- **3003 distinct circuits**, **3003 individually ODE-simulated** (1233 single-TF + 1770 multi-TF AND-logic); **215 on the Pareto front**; verdict **OK** cert **real**
- each circuit's Hill-ODE betas derive from its own parts (TF expression / promoter x enhancer / guide efficiency); robustness is a real per-circuit parameter sweep, knockdown a real steady-state readout.

| circuit | logic | efficacy | robustness | safety | TF knockdown | stable |
|---|---|---|---|---|---|---|
| CREB3L1 | neg-feedback | 0.95 | 1.00 | 0.90 | 0.68 | True |
| TCF7L2 | neg-feedback | 0.87 | 1.00 | 0.90 | 0.63 | True |
| HNF4G | neg-feedback | 0.94 | 1.00 | 0.90 | 0.62 | True |
| OVOL1 | neg-feedback | 0.96 | 1.00 | 0.90 | 0.62 | True |
| PLSCR1 | neg-feedback | 0.97 | 1.00 | 0.90 | 0.68 | True |
| ZNF823 | neg-feedback | 0.94 | 1.00 | 0.90 | 0.60 | True |
| SGSM2 | neg-feedback | 0.97 | 1.00 | 0.90 | 0.67 | True |
| FOXK2 | neg-feedback | 0.97 | 1.00 | 0.90 | 0.67 | True |
| HSF2 | neg-feedback | 0.96 | 1.00 | 0.90 | 0.62 | True |
| CDX2 | neg-feedback | 0.95 | 1.00 | 0.89 | 0.60 | True |

## Deep design — Basal subtype (individually-simulated circuits)

- **3003 distinct circuits**, **3003 individually ODE-simulated** (1233 single-TF + 1770 multi-TF AND-logic); **143 on the Pareto front**; verdict **OK** cert **real**
- each circuit's Hill-ODE betas derive from its own parts (TF expression / promoter x enhancer / guide efficiency); robustness is a real per-circuit parameter sweep, knockdown a real steady-state readout.

| circuit | logic | efficacy | robustness | safety | TF knockdown | stable |
|---|---|---|---|---|---|---|
| HMGA1 | neg-feedback | 0.97 | 1.00 | 0.86 | 0.69 | True |
| RARG | neg-feedback | 0.91 | 1.00 | 0.90 | 0.65 | True |
| FOXL1 | neg-feedback | 0.91 | 1.00 | 0.90 | 0.62 | True |
| PPARG | neg-feedback | 0.94 | 1.00 | 0.90 | 0.64 | True |
| FOXP4 | neg-feedback | 0.96 | 1.00 | 0.90 | 0.68 | True |
| TFAP2A | neg-feedback | 0.93 | 1.00 | 0.88 | 0.61 | True |
| FOSL1 | neg-feedback | 0.95 | 1.00 | 0.90 | 0.64 | True |
| OVOL1 | neg-feedback | 0.96 | 1.00 | 0.90 | 0.62 | True |
| ZNF618 | neg-feedback | 0.97 | 1.00 | 0.89 | 0.65 | True |
| ZNF554 | neg-feedback | 0.96 | 1.00 | 0.90 | 0.60 | True |

## PDAC chromatin acquired — the corpus is no longer healthy-only

The programme's central limitation was that **all 54 ChIP BAMs are healthy pancreas**, so the
PDAC-minus-healthy residual the design rests on could not be measured. ENCODE **Panc1** (a PDAC
cell line) supplies a **mark-matched panel on GRCh38** (`data/raw/encode-panc1-pdac`, 9 released
peak sets + signal bigWigs):

| PDAC mark | peaks | | PDAC mark | peaks |
|---|---|---|---|---|
| ATAC-seq | 99,761 | | H3K36me3 | 35,541 |
| CTCF | 58,083 | | H3K4me3 | 23,189 |
| H3K4me1 | 54,666 | | **TCF7L2** (a RAC target TF) | 10,794 |
| H3K27me3 | 46,565 | | H3K9me3 | 3,651 |
| **H3K27ac** | 45,824 | | | |

**First PDAC-chromatin validation of the RAC targets** (`scripts/pdac_vs_healthy_chromatin.py`,
`results/pdac_vs_healthy_chromatin.json`) — the targets were derived **without any PDAC chromatin**:

| test (Fisher, one-sided) | targets | background | odds ratio | p |
|---|---|---|---|---|
| accessible in PDAC (ATAC) | **94%** | 73% | **6.21** | **0.029** |
| active in PDAC (H3K27ac) | **89%** | 63% | **4.65** | **0.017** |

Both significant: the model's circuit targets sit in accessible, active PDAC chromatin.

### The disease residual, measured (signal-level, matched assay)

The peak-overlap contrast was unusable (see limit 2 below), so the residual was recomputed at
**signal level on matched assays**: ENCODE **signal p-value** tracks, PDAC (Panc1) vs healthy
pancreas, same assay/output type/assembly, log2((pdac+0.1)/(healthy+0.1)) over TSS +/- 2 kb
(`scripts/pdac_disease_residual.py`, `results/pdac_disease_residual_{ATAC-seq,H3K27ac}.json`;
~1,676 loci each). **No sealed study is touched** (`sealed_studies_touched: false`).

**Replicated across two independent marks:**

| mark | RAC targets | background | all loci | Mann-Whitney p |
|---|---|---|---|---|
| **ATAC-seq** (accessibility, vs 6 healthy tracks) | **+0.279** (70% up) | −0.263 (43%) | −0.256 | **0.010** |
| **H3K27ac** (active enhancer, vs 6 healthy tracks) | **+1.596** (80% up) | +0.261 (54%) | +0.277 | **0.00062** |

Both significant, on **independent marks**. The ATAC result is meaningful precisely because the
*overall* trend runs the other way — TF promoters are on average **less** accessible in Panc1 than in
healthy pancreas (−0.26; only 43% up) — yet **the RAC targets buck it**. On H3K27ac the effect is
larger still (+1.60 log2 ≈ **3x more active-enhancer signal** in PDAC than healthy). This is the
PDAC-minus-healthy residual the programme is built around, measured for the first time, and the
targets — derived with **no chromatin input at all** — are enriched in it on both marks.

**Honest limits — three of them.**
1. **Panc1 is a cultured PDAC cell line, not a primary tumour.** `pdac_tumor_bams` remains **0**;
   the registered benchmark asks for independent PDAC **tumour/organoid** studies, so this
   materially advances the disease-residual capability but does **not** discharge the benchmark.
2. **The peak-overlap differential was an artifact** (now superseded by the signal-level residual
   above). `pdac_specific_open` came out 0.00 because the healthy ATAC peak set has **874,795**
   peaks versus PDAC's **99,761** (~9x, a permissive merged set), making "open in PDAC, closed in
   healthy" empty by construction. Peak overlap cannot measure this contrast; signal can.
3. **TCGA has no PAAD ATAC-seq at all.** GDC carries ATAC-seq for **23 TCGA projects and PAAD is
   not among them** — Corces 2018 excluded pancreatic. Primary PDAC ATAC must come from GEO
   (87 candidate human PDAC ChIP/ATAC series exist), not TCGA. This is a hard fact, not a fetch failure.

**External corroboration of the top target.** Three 2024–25 GEO series are dedicated to *"KLF5
controls subtype-independent highly interactive enhancers in PDAC"* (GSE310807 / GSE309861 /
GSE309860). **KLF5** is exactly the TF the unsupervised attractor model ranked as the top
convergent target (DepMap essentiality 0.74) — independent literature agrees it is a genuine PDAC
enhancer master regulator.

## Stage 2 — the disease-residual objective is live (paired PDAC ↔ healthy)

The curriculum's Stage 2 (`progression_state_residual`) is what learns PDAC-minus-normal. It has
now been run on **genuinely paired** ENCODE Panc1 (PDAC) ↔ mark-matched healthy-pancreas shards,
initialised from the trained healthy prior (`models/chromatin/progression_paired_ckpt`).

| supervision | unpaired shards | **paired shards** |
|---|---|---|
| `residual_delta` (PDAC-minus-normal) | **0.000000** — silently inert | **+0.445470 — ACTIVE** |
| profile / correlation | 0.012 / 0.70 | 0.061 / 0.80 |

**This is the point of the whole programme:** the disease residual was previously unmeasurable
(healthy-only corpus) and then *silently no-op* (unpaired data reports `residual_delta = 0` while
appearing to train). It now fires on real paired data.

**Two real defects were found and are recorded, not papered over.**
1. **Unpaired data makes Stage 2 a no-op.** `paired_residual_loss` needs `paired_delta`; specs
   compiled as `pair_relation: unpaired` / `pair_group: ""` yield `residual_delta = 0` at every
   step while the run still reports a falling profile loss. A Stage-2 run can therefore *look*
   successful while learning no residual at all. Pairing requires a non-empty, identical
   `pair_group` plus exactly `state_reference` / `state_treatment` relations.
2. **`chromatin-compile` and `chromatin-pair` disagree on chromosome order.**
   Compile emits **lexicographic** order (chr1, chr10 … chr19, **chr2**, chr20 …);
   `pairing._coordinate_key` sorts **numerically** (chr2 → (0,2) precedes chr19 → (0,19)). The
   stream therefore goes *backwards* at chr19 → chr2 and pairing aborts with "shards are not
   coordinate-sorted". **The paired-state path could not run on ENCODE-compiled human data at
   all.** Fixed with a resort step (`scripts/resort_shards_for_pairing.py`) that reorders rows into
   `_coordinate_key` order and changes nothing else; frozen compile behaviour is untouched.

**Honest scope.** The paired Stage-2 run reached **250 steps** and then died at native level
(exit 127) under GPU contention from unrelated workloads on the same machine. So this is a
**proof that the residual objective is live and correctly wired**, *not* a converged Stage-2
model. Panc1 also remains a **cell line**, so this trains a PDAC-line residual, not a primary-tumour
one, and does not discharge the sealed benchmark.

Artifacts: `scripts/{make_panc1_track_specs,build_pdac_paired_specs,resort_shards_for_pairing}.py`,
`data/track_specs/{encode_panc1_pdac,pdac_paired}`, `data/processed/chromatin_pdac_paired_v1`
(40 paired shards, `valid: true`), `models/chromatin/progression_paired_ckpt`.

## CPTAC proteomics — protein-level TF activity (real mass-spec)

CPTAC PDAC (Cao 2021, `umich` pipeline) via the `cptac` package: **238 tumors x 11,419 proteins**
(`data/raw/cptac-pdac/cptac_pdac_proteome_umich.csv`). **651 of 1,639 Lambert TFs are quantified at
protein level — a 32x improvement on the 20-TF RPPA panel**, and it answers the protocol's
"confirm TF activity at the protein level (not just RNA)" requirement. Loader:
`load_cptac_pdac_proteome` (returns per-gene mean abundance **and detection rate**, because a TF
that is mRNA-high but protein-undetected is a poor CRISPRi target).

Applied to the RAC targets (12/20 have protein):

| gene | protein mean | detected in | reading |
|---|---|---|---|
| **AGR2** | **−0.489** | **100%** | reliably measured and **low** — converges with its hypermethylated promoter (beta 0.586): **two independent layers agree AGR2 is a poor target** despite 28% amplification |
| SETDB1 / SF3B1 / SMAD3 / ATM | ~0.0 to +0.07 | **100%** | robustly present |
| GATA6 | +0.057 | 67% | present |
| **FOSL1 / ZNF331** | — | **15% / 16%** | barely detectable at protein level -> weak evidence, deprioritise |
| BRCA2 | +0.029 | 38% | poorly detected |

## Hi-C 3D chromatin layer (real PANC-1, 4DN)

4DN experiment set **4DNESCCP4KTY** (Hi-C on PANC-1), hg38, using the **released derived tracks**
rather than the 2.6 GB contact matrix — exactly the protocol's "insulation scores ... A-Bin/B-Bin
scores ... eigenvector for activity" route: **A/B compartment eigenvector** (250 kb bins),
**diamond insulation score** (10 kb bins) and **called TAD boundaries**
(`scripts/hic_3d_layer.py`, `results/hic_3d_layer.json`; features for **1,676 genes**).

- **RAC targets: 90% sit in the active A compartment** (mean eigenvector 0.368) vs **70% of
  background TFs** (0.333).
- **Honest limit: this is a sanity check, not a discriminator.** Mann-Whitney on the continuous
  eigenvector is **p = 0.504** (null) — expressed TFs are *already* mostly in A compartments, so
  compartment membership adds no power beyond expression. Reported as a passing check, no claim.

## Single-cell PDAC — in-vivo malignant substrate (honest negative + a correction)

Real TISCH2-annotated PDAC single-cell data (`PAAD_CRA001160`, Peng 2019): **57,443 cells /
21,066 genes**, of which **11,401 are annotated malignant** (16,280 stromal, 11,654 immune).
Malignant cells only were pseudobulked per patient (**24 patients** with >= 30 malignant cells),
giving an ***in-vivo* tumour-intrinsic** regulatory substrate — no stroma, no immune, no culture
(`scripts/build_malignant_graph.py`, `results/single_cell_malignant_graph.json`).

| substrate | AUC core-essential | AUC PDAC-selective |
|---|---|---|
| cultured DepMap cell lines (RAC default) | **0.65–0.66** | 0.49 (null) |
| in-vivo malignant cells (this pull) | **0.51–0.55** | uninterpretable (n=3 positives) |

- **Real finding:** the **cultured cell-line graph beats the in-vivo tumour graph** for predicting
  CRISPR essentiality (0.65 vs 0.52). That is the readout matching the substrate — DepMap CRISPR is
  *measured in cell lines* — not a defect of the single-cell data.
- **Honest negative:** the in-vivo graph's apparent selective-dependency AUC (0.63–0.66) is **noise** —
  only **3** nodes pass `sel > 0.25`, and the value collapses to 0.43 at threshold 0.45. No claim made.
- **Correction to an earlier hypothesis.** It was proposed that the *composition confound* capped the
  PDAC-selective signal. **That is refuted.** A power audit over **1,164 TFs with DepMap CRISPR** finds
  only **7 PDAC-selective TFs at sel > 0.25** (5 at > 0.30; 30 even at a lenient > 0.10), versus 42
  core-essential at abs > 0.40. **PDAC-selective TF dependency is essentially nonexistent in DepMap**,
  so no graph substrate — bulk, cell-line, or in-vivo — can resolve it. This is a biological/power
  ceiling, and it means the selective-dependency question should not be pursued with this readout.

## DNA-methylation silencing filter (real TCGA-PAAD HM450)

Pulled via the cBioPortal generic-assay API: **396,065 HM450 CpG probes → 15,655 genes'
promoter probes** (TSS200 / TSS1500 / 1stExon / 5'UTR only), beta values across **183 tumors**
(`data/raw/tcga-paad/tcga_paad_methylation_promoter.csv`, `hm450_promoter_probe_map.json`).
Of 413 covered RAC nodes, **72 are hypermethylated** (beta > 0.5; median beta 0.085).

The layer answers the protocol's "silence false-positive open chromatin" requirement — an
ATAC-accessible promoter that is hypermethylated is a likely false-positive active element, and
an already-silenced promoter is a poor CRISPRi target. It is wired into the RAC graph
(`promoter_methylation`), the target table (`promoter_methylation_beta`,
`promoter_hypermethylated`, `accessible_but_silenced`) and the convergence score (silencing penalty).

Biology checks out:

| gene | promoter beta | reading |
|---|---|---|
| **TP63** | **0.733** | basal master TF **silenced** — consistent with classical-dominant TCGA-PAAD |
| **AGR2** | **0.586** | classical marker, hypermethylated -> **demoted** as a CRISPRi target despite 28% amplification |
| GATA6 / KLF5 / MYC / E2F1 / SETDB1 | 0.02–0.05 | unmethylated -> genuinely active; good CRISPRi targets |
| CDKN2A | 0.038 | **not** methylated — silenced by **deletion** (63% del) instead |

Applied to the top-20 RAC targets: **2 hypermethylated** (AGR2, FAM83A) flagged as poor CRISPRi
targets; **0 ATAC-open-but-silenced** false positives — the top targets are open *and*
unmethylated. **Correctness note:** HM450 `NAME`/`DESCRIPTION` fields are *positionally paired*;
an early mapping assigned a probe to every listed gene when *any* region was promoter-like,
which produced spurious hypermethylation calls (e.g. KRT19 beta 0.671 — implausible for a highly
expressed ductal marker). Fixed to strict positional pairing; KRT19/VIM correctly drop to
no-promoter-coverage and the real signals (TP63, AGR2, SETDB1) are unchanged.

## Module VIII — Regulatory Attractor Control (RAC)

A bistable graph dynamical system over the PDAC TF regulatory network, fit **only** to real
DepMap PDAC cell-state expression (54 lines), motif-directed by JASPAR (3,103 TF→target
promoter-motif edges), contextualised by ENCODE pancreas ATAC / H3K27ac, and annotated with
real TCGA-PAAD copy-number (GISTIC, 183 tumors). Node essentiality is the network-wide
*attractor collapse* induced by clamping each node down. DepMap CRISPR is **never used to fit
the dynamics** — it is a held-out, out-of-modality validation target.

- graph: **386 TF nodes, 7,492 co-expression edges, 3,103 motif-supported**, 54 PDAC lines,
  CNA on 385/386 nodes (80 amplified); bistable fixed-point error 0.030 (GPU).
- **Out-of-modality validation** (identify DepMap core-essential regulators, CRISPR held out):

  | essential threshold (−Chronos) | n positive | AUC collapse [95% CI] | AUC degree | AUC eigenvector | CI excludes chance |
  |---|---|---|---|---|---|
  | 0.3 | 34 | **0.661 [0.552, 0.765]** | 0.621 | 0.584 | yes |
  | 0.4 | 28 | **0.657 [0.533, 0.773]** | 0.624 | 0.584 | yes |
  | 0.5 | 16 | 0.569 [0.382, 0.750] | 0.533 | — | no (underpowered) |

  Permutation p (thr 0.4) = **0.006**. **8-member bootstrap ensemble** (refitting on resampled
  cell lines): AUC thr-0.4 mean **0.608 [0.539, 0.650]**, and **every one of the 8 members beats
  chance** (worst 0.532) — modest but robust. ~~Collapse beats degree and eigenvector centrality~~ **[RETRACTED — §1/§15]** at
  every threshold. **Honest limit:** the point estimate (0.66) is optimistic vs the resampled
  ensemble (0.61); the signal does **not** predict PDAC-*selective* dependency, and model
  regulatory coupling does **not** predict DepMap co-essentiality (tested, null).
- **Control design** (minimal repressible set moving the attractor along the GTEx-healthy
  direction, essential-safe): net healthy shift ~10–19; attractor mean shifts toward the dead basin.
- **Convergent circuit targets** (collapse + disease-up + master-regulator + motif + **CNA amplification** + driver):
  top hits include **GATA6** (classical-subtype master TF, 30% amplified), **KLF5** (lineage-survival
  TF, DepMap essential 0.74), **BRCA2 / SETDB1 / KMT2C** (IntOGen drivers), **E2F1** (proliferation),
  **AGR2** (classical marker, 28% amplified). That an unsupervised attractor model surfaces GATA6
  and KLF5 — the two textbook PDAC master TFs — near the top without seeing CRISPR data is the
  strongest single signal here.

### RAC definitive campaign (`results/rac_campaign.json`, scripts/heavy_rac_campaign.py)

A heavy sweep hardens the above. **Hyperparameter grid** (nodes 400/600/800 × co-expression
threshold 0.30/0.35/0.40, each an 8-member ensemble) finds a clear regime: **sparser, focused
graphs win** — best is **400 nodes / threshold 0.4** (ensemble AUC 0.61), and *adding* lower-variance
TFs (600–800 nodes) **lowers** AUC to 0.51–0.56 (collapse signal dilutes). At the best config:

- **Definitive:** ~~point AUC thr-0.4 **0.653 [0.539, 0.759]**, **50,000-permutation p = 0.0022**,
  beats degree (0.629) / eigenvector (0.584).~~ **[RETRACTED — §1/§15. The permutation tested against chance, not against degree, and the configuration was selected on the same CRISPR labels. Head-to-head: dAUC -0.082, partial rho 0.028 (p = 0.56).]** But the **40-member bootstrap ensemble** mean is
  **0.606 [0.475, 0.650]** and *not* every member beats chance — the point estimate is significant,
  the heavily-resampled signal is modest and noisy. Reported honestly, both ways.
- **Leave-cell-line-out CV** ~~(the strong result)~~ **[DEMOTED — leaked whole-panel statistics; internal diagnostic only]**: refit on 53 lines, the held-out real cell state
  is a **2× lower-residual fixed point than a permuted null (0.064 vs 0.128, Wilcoxon p ≈ 0)** —
  the learned regulatory dynamics **generalise to unseen PDAC cell lines**.
- **Convergent top-40** (with CNA amplification): **ELF3, KLF5** (ess 0.74), **GRHL2** (33% amp),
  **GATA6** (classical, 30% amp), **AGR2, SETDB1**.

Artifacts: `results/attractor_{map,validation,control,targets}.json`, `results/rac_campaign.json`,
`figures/fig_attractor.png`. CLI: `pdac attractor-design`.

## Long-range PDACircuitFormer — first real training

The 196,608 bp model (~1.95M trainable params) is **trained from scratch** on all 33,156 compiled
healthy-pancreas shards (Stage 1 healthy prior), not just forward-checked. Held-out validation
profile loss 0.0088 → **0.0079** at ~step 700, then plateau (converges in ~15 min; the healthy
window set is a single ~2,600-window pass — `results/chromatin_training_curve.json`,
`models/chromatin/healthy_prior_ckpt`).

**Multi-seed reproducibility** (`results/chromatin_ensemble.json`, `figures/fig_chromatin_training.png`) —
independent random-seed retrainings on the real shards:

| n seeds | held-out profile correlation | std | min | max |
|---|---|---|---|---|
| 8 | **0.7102** | 0.0091 | 0.698 | 0.728 |

The trained long-range model predicts held-out healthy-pancreas chromatin at **r ≈ 0.71 ± 0.01**,
reproducibly across 8 independent seeds (the estimate converged: seeds 7→8 moved the mean by 0.0007). Honest scope unchanged: it learns the healthy *counterfactual prior*
well, but the "beats-Enformer" benchmark still **ABSTAINs** until PDAC tumor/perturbation
chromatin and hash-locked baselines exist. (A 24-seed target was truncated at 8 — the
environment terminates long background jobs; the ensemble script resumes, and the std is already
~1.4% relative, so additional seeds would not move the estimate.)

## Honest scope

- **RAC dynamics are fit to expression only; DepMap CRISPR is held out** — the AUC 0.65–0.67 is a genuine out-of-modality check, not a re-fit. **[RETRACTED — §1/§15: out-of-modality is true but irrelevant; the configuration was selected on the same CRISPR labels, and against the degree baseline the gain is dAUC -0.082.]**
- RAC essentiality signal is modest (AUC ≈ 0.67) and does not resolve lineage-*selective* dependency; treat convergent targets as hypotheses, not validated dependencies.
- Tumor-vs-normal specificity is cross-platform (TCGA RSEM vs GTEx TPM) — down-weighted.
- Off-target search is a bounded seed-and-extend; exhaustive genome-wide is documented, not run.
- Immunogenicity is a coarse heuristic proxy with wide uncertainty, never a clinical prediction.
- CFD uses position-weighted (Hsu-2013) penalties; the exact Doench-2016 per-nucleotide matrix is approximated.
- Generated promoters are de-novo candidates, never treated as real data.
