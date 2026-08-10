---
layout: default
title: "Addendum: chromatin"
subtitle: "The H3K27ac residual analysis, its controls, and what ATAC does and does not replicate."
description: "The H3K27ac residual analysis, its controls, and what ATAC does and does not replicate."
permalink: /addenda/chromatin/
group: addenda
order: 2
---

This addendum consolidates the single claim in this project that survived the
full review arc, states exactly what it does and does not support, and records the complete evidence
so it can be checked rather than trusted. Every other substantive claim was retracted, retired,
shown unrescuable, or exposed as confounded — see [ADDENDUM_DYNAMICS.md]({{ '/addenda/dynamics/' | relative_url }}),
[ADDENDUM_RAC_V2.md]({{ '/addenda/rac-v2/' | relative_url }}) and [../REVIEW_RESPONSE.md]({{ '/reports/review/' | relative_url }}).

Sections §25–§28 of the review response are the primary record. Scripts:
`fetch_foldchange_tracks.py`, `pdac_residual_foldchange.py`, `h3k27ac_fragility.py`,
`h3k27ac_window_and_loci.py`, `h3k27ac_pseudocount.py`.

---

## 1. The claim, stated precisely

> The 20 transcription factors surfaced by RAC sit on **PDAC-gained H3K27ac at their promoters**,
> relative to healthy pancreas, and this is not explained by their expression level, by the
> disease-expression change they were selected on, or by their network hub-ness.

Deliberately **not** claimed: that the effect is large, that it extends beyond the promoter, that it
generalises to open chromatin, that it validates RAC, or that it identifies therapeutic targets.

---

## 2. Why the original version of this number was untrustworthy

The published figure (targets +1.596 vs background +0.261, p = 0.00062) had two independent defects,
and the second was worse than the first:

1. **Normalisation.** It used ENCODE **signal p-value** tracks, which conflate sequencing depth with
   enrichment — a deeper library yields larger p-values at the same true enrichment.
2. **Background.** It compared targets against *all* other TFs. RAC targets are high-expression,
   high-degree hubs, and a highly expressed gene has acetylated promoter chromatin close to
   definitionally. The contrast risked measuring **expression**, which is precisely the confound that
   destroyed the supervised selective signal in ADDENDUM_RAC_V2 §9.

---

## 3. What was done

**Fold-change tracks, matched by processing run.** Four ENCODE fold-change-over-control bigWigs
(3.5 GB, sha256 in `data/manifests/encode-foldchange.json`). Files were chosen by matching the
`derived_from` **processing run** of the signal p-value tracks they replace — Panc1 H3K27ac
`ENCFF528UFR → ENCFF047WWJ` (both from `ENCFF384KMQ + ENCFF675MQQ`), Panc1 ATAC
`ENCFF055ZEE → ENCFF174PXJ` (both from `ENCFF836WDC`) — so the old-vs-new comparison changes
normalisation **only**. `ENCSR000EXK` contains a decoy second run (`ENCFF240BXE`) from different
alignments; selecting it would have changed two things at once.

**Three matched backgrounds.** Each target paired with up to 3 background genes within a 0.25 SD
caliper on absolute PDAC expression, on `disease_log2fc` (the RAC selection variable), and on
co-expression degree.

---

## 4. The complete evidence

### 4.1 Matched controls (§25)

| contrast | matched score (tgt vs bg) | targets | background | MWU p |
|---|---|---|---|---|
| all background | — | +0.919 | −0.091 | **0.0022** |
| absolute PDAC expression | 2.232 vs 2.230 | +0.919 | +0.105 | **0.017** |
| `disease_log2fc` — circularity | 6.867 vs 6.75 | +0.919 | +0.022 | **0.025** |
| co-expression degree — hub-ness | 136.8 vs 136.75 | +0.919 | −0.017 | **0.010** |

The `disease_log2fc` control is the one that matters. RAC selected targets partly on a
TCGA-vs-GTEx differential, and this residual is a Panc1-vs-healthy-pancreas contrast — both are
"PDAC vs normal pancreas", so a disease-up gene would have disease-up promoter acetylation almost by
construction. Matching on absolute expression does **not** control for this: it matches *level*, not
*change*. Matching on `disease_log2fc` does.

### 4.2 Fragility (§26)

- **Set-level permutation, p = 0.00155.** Mann-Whitney treats genes as exchangeable individuals, but
  the claim is about a *set of 20*. Drawing 20 random background genes 20,000 times: observed +0.919
  against null mean −0.093, null 95th percentile +0.461. The stricter, more appropriate null is
  **more** significant than the gene-wise test.
- **Leave-one-target-out, worst p = 0.0058** (dropping HOXA3, the largest contributor). Not one gene.
- **Bootstrap 95% CI [0.244, 1.679]** — excludes zero, and wide.
- **Caliper sensitivity flat** across 0.10 / 0.25 / 0.50 SD on all three controls.

### 4.3 Window (§27) and pseudocount (§28)

| parameter | range | significant | fold-change span |
|---|---|---|---|
| window | ±500 bp – ±25 kb | **6/6** | 1.01× – 2.05× |
| pseudocount | 0.01 – 2.0 | **6/6** | 1.45× – 2.15× |

**12 parameter settings, no exceptions.** Significance is essentially immune to the pseudocount
(MWU p 0.0021–0.0032 across a 200× range) because both tests are **rank-based** and changing the
constant is nearly a monotone transform, so gene ordering barely moves. The rank test carries the
claim and is unaffected; the mean carries effect size and is not.

### 4.4 Two corrections the aggregate was hiding

- **Promoter-local, not domain-wide.** Absolute target enrichment collapses from 2.05× (±500 bp) to
  **1.01× (±25 kb)** — no gain at all at domain scale. Significance at wide windows comes from the
  *background* becoming depleted (−0.533), not from targets staying high. Targets **gain**
  acetylation at promoter scale and merely **retain** it where other TF loci lose it.
- **The top locus is substantially a pseudocount artifact.** HOXA3's healthy fold-change is exactly
  0.000, so its 58.7× ratio is `log2(5.872 / 0.1)` — set by the constant, not the data. It is the
  only target with near-zero healthy signal (the other 19 have real signal on both sides).

### 4.5 An unplanned coherence check that passes

**GATA6 is the most negative target (−2.24, 0.21×).** GATA6 is the classical-identity factor, and §9
independently established that PANC-1 sits *below* the panel mean on the classical programme. A
classical-identity enhancer should be less acetylated in a non-classical line — and it is. Nobody
designed this check; it fell out of the per-locus table.

---

## 5. Effect size — reported as a range, because it is not well determined

| estimator | at published settings | span across all 12 settings |
|---|---|---|
| raw mean | +0.919 (1.89×) | +0.532 … +1.101 |
| **median** | **+0.852 (1.81×)** | +0.564 … +0.891 |
| **mean excl. zero-denominator** | **+0.659 (1.58×)** | +0.457 … +0.676 |

**The defensible statement is 1.5–1.8×, and the honest one adds that no single number is
well-supported.** The existence of the effect is robust across every setting tested; its magnitude
moves by roughly 2× under constants that no data constrains. Quoting "1.89×" as *the* effect size
would misrepresent it.

---

## 6. Limitations, none of which the evidence removes

- **PANC-1 is an intermediate substrate.** §9 showed it sits below the panel mean on *both* Moffitt
  programmes — it represents neither the basal nor the classical target set. This is a cell line
  standing in for a tumour, and a poorly-typed one.
- **One healthy fold-change track per mark**, versus up to six averaged in the original p-value run.
  The healthy reference is noisier; this is the more conservative test but also the thinner one.
- **n = 20 targets.** The bootstrap CI [0.244, 1.679] reflects that honestly.
- **ATAC does not replicate** (primary contrast p = 0.074). This is an **H3K27ac-specific**
  observation, not a general chromatin claim. Its matched controls were nominally significant and are
  deliberately *not* used to rescue it — significant subsets do not save a failed primary contrast.
- **It does not resurrect RAC.** The targets come from a model whose essentiality claim is retracted
  (§15), whose bistable framing is retired (§17), and which is unrescuable by gain (§18) or substrate
  (§19). This says the *gene set* has a chromatin property; it says nothing about collapse predicting
  essentiality.

---

## 7. What would actually confirm it

1. **A properly-typed substrate** — primary PDAC tumour or organoid H3K27ac, or cell lines at the
   poles of the Moffitt signature, rather than an intermediate line.
2. **Multiple healthy references**, to replace a single track and give the residual an error bar.
3. **A second mark that does replicate** — ATAC did not here, so whether this is acetylation-specific
   or a general open-chromatin phenomenon is unresolved.
4. **An independent target set.** These 20 genes came from a retracted model; whether the property
   attaches to *these* genes or to PDAC-relevant TFs generally cannot be told from one set.

Until then the claim stands as written in §1: real, promoter-local, modest, and narrow.
