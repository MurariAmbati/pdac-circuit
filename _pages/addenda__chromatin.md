---
layout: default
title: "Addendum: chromatin"
subtitle: "The H3K27ac residual analysis, its controls, and what ATAC does and does not replicate."
description: "The H3K27ac residual analysis, its controls, and what ATAC does and does not replicate."
permalink: /addenda/chromatin/
group: addenda
order: 2
---

One claim in this project survived the full review arc. This is it. Everything else was retracted,
retired, shown unrescuable, or caught out as confounded, which the [dynamics addendum]({{ '/addenda/dynamics/' | relative_url }}),
the [attractor-model rebuild]({{ '/addenda/rac-v2/' | relative_url }}) and the [review arc]({{ '/reports/review/' | relative_url }}) cover.
Sections §25–§28 of the review response hold the working record, built by
`fetch_foldchange_tracks.py`, `pdac_residual_foldchange.py`, `h3k27ac_fragility.py`,
`h3k27ac_window_and_loci.py` and `h3k27ac_pseudocount.py`.

---

## 1. The claim

> The 20 transcription factors RAC surfaced carry PDAC-gained H3K27ac at their promoters, measured
> against healthy pancreas. Expression level does not explain it. Neither does the disease-expression
> change they were picked on, nor how well connected they are.

Not claimed: that the effect is big, that it reaches past the promoter, that it carries over to open
chromatin, that it rescues RAC, or that these are drug targets.

---

## 2. Why the first number was no good

The published figure (targets +1.596, background +0.261, p = 0.00062) had two problems.

It used ENCODE signal p-value tracks. Those mix sequencing depth into enrichment, so a deeper library
posts a bigger p-value at identical true signal.

Worse, it set targets against every other transcription factor. RAC targets are highly expressed and
heavily connected, and a busy gene has acetylated promoter chromatin almost by definition. So the
contrast was at risk of measuring expression, the same confound that wrecked the supervised selective
signal in ADDENDUM_RAC_V2 §8.

---

## 3. What replaced it

Four ENCODE fold-change-over-control bigWigs stood in for the p-value tracks (3.5 GB, sha256 in
`data/manifests/encode-foldchange.json`). Each was picked to share the `derived_from` processing run
of the track it replaces, so normalisation is the only thing that moves. `ENCSR000EXK` carries a decoy
second run off different alignments; taking it would have shifted two things at once.

Every target then got up to three background genes inside a 0.25 SD caliper on absolute PDAC
expression, on `disease_log2fc` (the variable RAC selected on), and on co-expression degree.

---

## 4. The evidence

### 4.1 Matched controls (§25)

| contrast | matched score (tgt vs bg) | targets | background | MWU p |
|---|---|---|---|---|
| all background | n/a | +0.919 | −0.091 | **0.0022** |
| absolute PDAC expression | 2.232 vs 2.230 | +0.919 | +0.105 | **0.017** |
| `disease_log2fc`, circularity | 6.867 vs 6.75 | +0.919 | +0.022 | **0.025** |
| co-expression degree, hub-ness | 136.8 vs 136.75 | +0.919 | −0.017 | **0.010** |

The `disease_log2fc` row is the one that counts. RAC picked targets partly on a TCGA-against-GTEx
differential, and this residual is Panc1 against healthy pancreas. Both are disease against normal
tissue, so a disease-up gene arrives with disease-up promoter acetylation almost for free. Matching on
absolute expression will not catch that, since it pins level and leaves change free. Matching on
`disease_log2fc` pins the change.

### 4.2 Fragility (§26)

Mann-Whitney treats genes as swappable individuals, but the claim covers a set of 20. Draw 20 random
background genes 20,000 times and the observed +0.919 lands against a null mean of −0.093 and a null
95th percentile of +0.461, giving p = 0.00155. The stricter test is the more significant one.

Leave-one-target-out bottoms out at p = 0.0058 when HOXA3 goes, so no single gene is holding this up.
The bootstrap 95 per cent interval, [0.244, 1.679], clears zero and is wide. Caliper sensitivity stays
flat at 0.10, 0.25 and 0.50 SD.

### 4.3 Window (§27) and pseudocount (§28)

| parameter | range | significant | fold-change span |
|---|---|---|---|
| window | ±500 bp to ±25 kb | **6/6** | 1.01× to 2.05× |
| pseudocount | 0.01 to 2.0 | **6/6** | 1.45× to 2.15× |

Twelve settings, no exceptions. The pseudocount barely touches significance, MWU p moving only from
0.0021 to 0.0032 over a 200-fold sweep, because both tests work on ranks and shifting the constant is
close to a monotone transform, so gene order hardly budges. Ranks carry the claim and hold. The mean
carries effect size and does not.

### 4.4 Two things the aggregate hid

The gain is promoter-local, not domain-wide. Target enrichment falls from 2.05× at ±500 bp to 1.01× at
±25 kb, so at domain scale there is nothing. What makes wide windows significant is the background
going depleted (−0.533), not targets holding high. Targets pick up acetylation at promoter scale and
simply keep it where other transcription-factor loci shed theirs.

The top locus is largely an artifact of the pseudocount. HOXA3's healthy fold-change is exactly 0.000,
so its 58.7× ratio is `log2(5.872 / 0.1)`, fixed by the constant and not by data. It is the only
target with near-zero healthy signal; the other 19 have real signal both sides.

### 4.5 A coherence check nobody planned

GATA6 is the most negative target, −2.24 or 0.21×. It is the classical-identity factor, and §9 of the
review arc separately put PANC-1 below the panel mean on the classical programme. A classical-identity
enhancer ought to be less acetylated in that line. Nobody set this check up. It fell out of the
per-locus table.

---

## 5. Effect size, as a range

| estimator | at published settings | span across all 12 settings |
|---|---|---|
| raw mean | +0.919 (1.89×) | +0.532 to +1.101 |
| **median** | **+0.852 (1.81×)** | +0.564 to +0.891 |
| **mean excluding zero denominator** | **+0.659 (1.58×)** | +0.457 to +0.676 |

Defensible: 1.5× to 1.8×. The fuller answer adds that no single figure is well supported. The effect
shows up under every setting tried, while its size swings about twofold on constants that no data
pins down. Quoting 1.89× as the effect size would oversell it.

---

## 6. Limits the evidence does not clear

PANC-1 is an awkward substrate. §9 of the review arc put it below the panel mean on both Moffitt
programmes, so it stands for neither the basal nor the classical target set. A cell line is standing
in for a tumour here, and a badly typed one.

One healthy fold-change track per mark, against up to six averaged in the original p-value run. The
healthy reference is noisier. That makes this the more conservative test and also the thinner one.

Twenty targets. The bootstrap interval [0.244, 1.679] owns up to that.

ATAC does not replicate, primary contrast p = 0.074. So this is specific to H3K27ac and is no general
chromatin claim. Its matched controls came in nominally significant and are deliberately left out of
the argument, because a significant subset cannot save a failed primary contrast.

None of it revives RAC. These targets come from a model whose essentiality claim is retracted (§15),
whose bistable framing is retired (§17), and which no gain (§18) or substrate (§19) could rescue. The
gene set has a chromatin property. Collapse still predicts nothing.

---

## 7. What would settle it

A properly typed substrate, meaning primary tumour or organoid H3K27ac in place of an awkward line.
Several healthy references, to put an error bar on the residual. A second mark that replicates, since
ATAC did not and nobody yet knows whether this is acetylation-specific. And an independent target set,
because these 20 genes came out of a retracted model.

Short of that, the claim stands as §1 puts it. Real, promoter-local, modest, narrow.
