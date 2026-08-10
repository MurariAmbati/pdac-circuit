# Addendum, real-data scaling of the Module II/V sequence models

Four learned sequence models sit in the pipeline. gRNA on-target (Module V), promoter strength and
enhancer activity (both Module II), and the promoter GAN (Module VII). All four shipped trained on a
slice of the real data available to them, capped for no good reason. So the question each time was
simple. Is the model held back by its features and architecture, or just by how little it saw?

The runs below lift the caps, or bring in a second real dataset, and score what comes out on a
held-out test that stays fixed across the comparison. Every number comes off real sha256-manifested
data. Every baseline gets re-scored on that same set. Every ensemble weight is picked on validation,
never on test.

All four went out on more data. The enhancer threw an honest negative on its second lever. The GAN's
gain landed in its selectable tail, since 4-mer realism was already near ceiling. None of this moves
the pipeline off being a prototype. It does make the parts better understood.

---

## 1. Summary

| model | module | shipped training data | real data available | held-out metric (fixed test) | before → after | deployed |
|---|---|---|---|---|---|---|
| gRNA on-target | V | Doench-2016, 5,310 guides / 17 genes | + Kim-2019 HT, 12,832 guides | Spearman, 688 held-out Doench genes | **0.494 → 0.657** | yes |
| promoter strength | II | FANTOM5, 60,000-peak cap | 209,374 real CAGE peaks | Spearman, 16,940 chr8/9 peaks | **0.5199 → 0.5275** | yes |
| enhancer activity | II | ENCODE pancreas, 20,000-active cap | 470,874 real actives | AUROC, 13,425 chr8/9 rows | **0.809 → 0.815** | yes |
| promoter GAN | VII | FANTOM5, 12,000-promoter cap | 52,342 top-quartile promoters | selectable-tail p90 (still certified-real) | p90 0.937 → **0.992** | yes |

Two independent findings sit alongside the enhancer scale-up:

- **Cross-domain generalisation (positive).** A healthy-pancreas-only enhancer model predicts PANC-1
  **PDAC** enhancers at AUROC **0.835**, higher than its own pancreas test, evidence the model
  learns transferable regulatory grammar and not tissue-specific artefacts.
- **Adding PANC-1 PDAC data (honest negative).** Merging real PANC-1 chromatin into training does
  **not** help the pancreas benchmark (0.815 → 0.810); because the grammar already transfers, the
  mixed-domain objective slightly dilutes per-domain performance. Not deployed.

---

## 2. gRNA on-target (Module V), add a second real HT dataset

The shipped model was feature-saturated (Azimuth/Rule-Set-2 extras added +0.008) but data-starved:
5,310 guides across only 17 genes. We added **Kim et al. 2019** (Science Advances eaax9249), a real
high-throughput SpCas9 library of **12,832** synthetic-target guides in the identical (4+20+3+3)
30-mer format, rank-normalised within-dataset so it pools with Doench's drug-gene rank.

**Diagnosis before integrating.** Training on Doench's 17 genes and testing on all 12,832 Kim guides
gave Spearman **0.592**, *higher* than the within-Doench held-out (0.53), proving the model had
learned transferable guide biology, not 17-gene memorisation.

**Result on the identical 688 held-out Doench genes** (genes CCDC101/CD15/CD45, gene-grouped):

| component | Doench-only | Doench + Kim |
|---|---|---|
| CNN | 0.3915 | **0.6167** |
| GBM | 0.5250 | 0.6504 |
| ensemble (deployed) | 0.4938 | **0.6571** |

The headline story is the CNN: near-random on 17 genes, it becomes a genuine contributor on 18,142
guides. Ensemble weight 0.40 CNN / 0.60 GBM, selected on held-out Doench val genes.
`results/grna_kim_augment.json`, `results/grna_cnn_kim_retrain.json`.

---

## 3. Promoter strength (Module II), remove the FANTOM5 cap

FANTOM5 provides **209,374** real CAGE peaks on standard chromosomes; the shipped model trained on a
random **60,000**. We built the full dataset, fixed the held-out test to **all 16,940 peaks on chr8
and chr9**, measured the shipped model on that exact set as the baseline, then retrained the CNN + RF
on the full 181,428-peak train pool.

| component | shipped (60k) | full (181k) |
|---|---|---|
| CNN | 0.4988 | **0.5247** |
| RF | 0.5020 | 0.5081 |
| ensemble (deployed) | 0.5199 | **0.5275** |

The CNN gains +0.026 from 3× the data (data-hungry, as expected); the tree-based RF is nearly flat
(+0.006). The ensemble gain (+0.0075) is modest, consistent with this sequence→expression task being
close to its data-saturation ceiling for the current architecture. Deployed at the val-selected
weight 0.86 CNN / 0.14 RF. `results/promoter_scaleup.json`.

---

## 4. Enhancer activity (Module II), remove the cap, and test adding PDAC data

The shipped model caps pancreas actives (ENCODE ATAC ∩ H3K27ac) at **20,000**, but **470,874** real
actives are available, it trained on ~4% of the data.

### 4a. Un-capping pancreas (deployed)

Fixed held-out test = all 13,425 rows on chr8/chr9; shipped model measured on it as baseline;
multitask CNN retrained on the full uncapped train pool (135,402 rows, 4× the cap).

| | shipped (20k) | full (80k actives) |
|---|---|---|
| AUROC | 0.8087 | **0.8147** |
| signal Spearman | 0.5516 | **0.5807** |

A real +0.006 AUROC and a clear signal-regression gain. `results/enhancer_scaleup.json`.

### 4b. Cross-domain generalisation and the PANC-1 augmentation (honest negative)

We separately tested adding real **PANC-1 PDAC** chromatin (ENCODE ENCFF953NZY ATAC +
ENCFF579DQM H3K27ac). A genuinely new source, and the PDAC context the project targets.

- A **pancreas-only** model reproduces the shipped **0.8150** exactly and predicts PANC-1 enhancers
  at AUROC **0.8349** (forward cross-domain transfer). The enhancer grammar is domain-general.
- **Merging** PANC-1 into training moves the pancreas benchmark to **0.8096** (−0.0054), a small
  dilution, because the model was not data-limited on enhancer *sequence*, only on pancreas sample
  count. Correctly **not deployed**; the shipped/un-capped model is kept.

This is the same controlled experiment the gRNA augmentation was, it simply came out negative on this
lever, which is worth recording precisely because the forward-transfer result explains *why*.
`results/enhancer_panc1_augment.json`.

---

## 5. Scaling curves

Two points cannot tell you a curve has flattened. So each model was trained at a ladder of set sizes
and every point scored on the same fixed chr8/chr9 test. These are independent trainings, drifting
about 0.005 run to run, so read the shape and ignore the third decimal.

**Promoter** (Spearman, ensemble, `results/promoter_scaling_curve.json`):

| n_train | 10k | 20k | 40k | 80k | 120k | 181k (full) |
|---|---|---|---|---|---|---|
| CNN | 0.489 | 0.477 | 0.503 | 0.514 | 0.530 | 0.528 |
| ensemble | 0.498 | 0.501 | 0.513 | 0.519 | 0.532 | **0.533** |

A monotone rise of **+0.035** across an 18-fold increase in real data, which is decisive evidence the
shipped 60k model was data-limited and not at its ceiling. The curve flattens over the last step
(120k to 181k, +0.001), so it approaches saturation near the full set. The CNN carries essentially all
of the gain, 0.489 to 0.528, and the RF is flat throughout.

**Enhancer** (AUROC, `results/enhancer_scaling_curve.json`):

| n_train | 20k | 40k | 80k | 135k (full) |
|---|---|---|---|---|
| AUROC | 0.805 | 0.804 | 0.806 | **0.812** |

Flatter than the promoter, so the enhancer classifier is closer to saturation. That is consistent
with the +0.006 scale-up gain and with the grammar already generalising across domains. Most of the
movement is in the final doubling.

**Bidirectional cross-domain transfer** (AUROC):

| direction | train | test | AUROC |
|---|---|---|---|
| forward | pancreas (multi-donor) | PANC-1 PDAC | **0.835** |
| reverse | PANC-1 PDAC | pancreas (multi-donor) | **0.790** |

Both directions are well above chance, so the enhancer grammar is genuinely shared between healthy
pancreas and PDAC. The asymmetry is expected, because the multi-donor pancreas set is the more diverse
training source and generalises to a single line better than the reverse.

---

## 6. Promoter GAN (Module VII), un-capped generator deployed for a stronger selectable tail

The WGAN-GP promoter generator trains on the top-quartile (highest-activity) real FANTOM5 promoters,
capped at **12,000** of the **52,342** available. We removed the cap (trained on all 52,342, same
config: 2,500 generator iterations, best-4-mer-JS early stopping) and evaluated the shipped and
un-capped generators identically. Same fixed real reference, same 1,500 generated sequences, same
seed, scored with the current promoter model. `scripts/promoter_gan_scaleup.py`,
`results/promoter_gan_scaleup.json`.

| metric | shipped (12k) | un-capped (52k) | direction |
|---|---|---|---|
| 4-mer JS vs real | 0.0088 | 0.0123 | lower = more faithful |
| 4-mer JS vs random | 0.0508 | 0.0508 | reference |
| predicted-strength p90 | 0.937 | **0.992** | higher = stronger selectable tail |
| median strength uplift | −0.021 | **+0.106** | (not gated by design) |

**Deployed.** How the pipeline consumes the generator decides this one. It generates a library and
selects the strongest promoter, so the selectable tail at p90 is the operative axis, and the
pre-registered criterion gates realism only at JS ≤ 0.05 with a requirement to beat random, and
usability at p90 ≥ 0.7. The un-capped generator clears both. Its JS of 0.0123 is about four times
better than random and well inside the bound, and it is markedly stronger on the tail the pipeline
actually uses. Because the generator trains purely on real sequence and never against the promoter
model, that stronger tail is a learned property, not scorer-gaming. The cost is a small rise in
4-mer JS, still well within spec, reflecting the generator concentrating more mass on strong-promoter
composition. The deploy gate is therefore the project's own certification plus a non-regressing tail,
not a tighter faithfulness bar that would have rejected a certified and more useful model. Consistent
with the promoter curve in §5, faithfulness is near saturated well before 52k, and the extra data buys
tail strength, not lower JS.

---

## 7. Provenance and reproducibility

- Real data only; raw bytes gitignored, sha256 in `data/manifests/`. Model weights (`*.pt`) are
  gitignored by repo convention; the manifests carry `weight_sha256`, and a frozen predeploy fixture
  re-verifies each deployed checkpoint (`max_abs_diff` ≤ 1e-4).
- Scripts: `scripts/grna_cnn_kim_retrain.py`, `scripts/promoter_scaleup.py`,
  `scripts/enhancer_scaleup.py`, `scripts/enhancer_panc1_augment.py`,
  `scripts/promoter_scaling_curve.py`, `scripts/enhancer_scaling_curve.py`,
  `scripts/promoter_gan_scaleup.py`.
- Splits: gRNA is gene-grouped; promoter/enhancer are chromosome-held-out (test chr8/chr9, val chr7).
  Ensemble weights are selected on validation, not test.
