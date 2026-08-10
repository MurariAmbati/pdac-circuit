# Addendum: PDAC-gained promoter H3K27ac at RAC-surfaced loci

This is the one claim in the project that survived the full review arc. Every other substantive claim
was retracted, retired, shown unrescuable, or exposed as confounded, as recorded in the
[dynamics addendum](ADDENDUM_DYNAMICS.md), the [attractor-model rebuild](ADDENDUM_RAC_V2.md) and the
[review arc](../REVIEW_RESPONSE.md). Sections §25–§28 of the review response are the primary
record, produced by `fetch_foldchange_tracks.py`, `pdac_residual_foldchange.py`,
`h3k27ac_fragility.py`, `h3k27ac_window_and_loci.py` and `h3k27ac_pseudocount.py`.

---

## 1. The claim, stated precisely

> The 20 transcription factors surfaced by RAC sit on PDAC-gained H3K27ac at their promoters, relative
> to healthy pancreas, and this is not explained by their expression level, by the disease-expression
> change they were selected on, or by their network hub-ness.

Deliberately not claimed: that the effect is large, that it extends beyond the promoter, that it
generalises to open chromatin, that it validates RAC, or that it identifies therapeutic targets.

---

## 2. Why the original number was untrustworthy

The published figure (targets +1.596 against background +0.261, p = 0.00062) had two defects. It used
ENCODE signal p-value tracks, which conflate sequencing depth with enrichment, so a deeper library
yields larger p-values at the same true enrichment. Worse, it compared targets against all other
transcription factors. RAC targets are high-expression, high-degree hubs, and a highly expressed gene
has acetylated promoter chromatin close to definitionally, so the contrast risked measuring expression.
That is the confound which destroyed the supervised selective signal in ADDENDUM_RAC_V2 §9.

---

## 3. What was done

Four ENCODE fold-change-over-control bigWigs replaced the p-value tracks (3.5 GB, sha256 in
`data/manifests/encode-foldchange.json`), matched on the `derived_from` processing run of the tracks
they replace, so the comparison changes normalisation only. `ENCSR000EXK` contains a decoy second run
from different alignments, and selecting it would have changed two things at once. Each target was then
paired with up to three background genes inside a 0.25 SD caliper on absolute PDAC expression, on
`disease_log2fc` (the RAC selection variable), and on co-expression degree.

---

## 4. The evidence

### 4.1 Matched controls (§25)

| contrast | matched score (tgt vs bg) | targets | background | MWU p |
|---|---|---|---|---|
| all background | n/a | +0.919 | −0.091 | **0.0022** |
| absolute PDAC expression | 2.232 vs 2.230 | +0.919 | +0.105 | **0.017** |
| `disease_log2fc`, circularity | 6.867 vs 6.75 | +0.919 | +0.022 | **0.025** |
| co-expression degree, hub-ness | 136.8 vs 136.75 | +0.919 | −0.017 | **0.010** |

The `disease_log2fc` control is the one that matters. RAC selected targets partly on a TCGA-against-GTEx
differential, and this residual is a Panc1-against-healthy-pancreas contrast. Both are PDAC against
normal pancreas, so a disease-up gene would have disease-up promoter acetylation almost by
construction. Matching on absolute expression does not control for this, because it matches level
rather than change. Matching on `disease_log2fc` does.

### 4.2 Fragility (§26)

Mann-Whitney treats genes as exchangeable individuals, but the claim is about a set of 20. Drawing 20
random background genes 20,000 times puts the observed +0.919 against a null mean of −0.093 and a null
95th percentile of +0.461, giving p = 0.00155, so the stricter null is more significant than the
gene-wise test. Leave-one-target-out gives a worst p of 0.0058 when HOXA3 is dropped, so the result is
not one gene. The bootstrap 95 per cent interval is [0.244, 1.679], which excludes zero and is wide,
and caliper sensitivity is flat across 0.10, 0.25 and 0.50 SD.

### 4.3 Window (§27) and pseudocount (§28)

| parameter | range | significant | fold-change span |
|---|---|---|---|
| window | ±500 bp to ±25 kb | **6/6** | 1.01× to 2.05× |
| pseudocount | 0.01 to 2.0 | **6/6** | 1.45× to 2.15× |

Twelve parameter settings, no exceptions. Significance is essentially immune to the pseudocount, with
MWU p from 0.0021 to 0.0032 across a 200-fold range, because both tests are rank-based and changing the
constant is nearly a monotone transform, so gene ordering barely moves. The rank test carries the claim
and is unaffected. The mean carries effect size and is not.

### 4.4 Two corrections the aggregate was hiding

The effect is promoter-local rather than domain-wide. Absolute target enrichment collapses from 2.05×
at ±500 bp to 1.01× at ±25 kb, meaning no gain at all at domain scale. Significance at wide windows
comes from the background becoming depleted (−0.533), not from targets staying high. Targets gain
acetylation at promoter scale and merely retain it where other transcription-factor loci lose it.

The top locus is also substantially a pseudocount artifact. HOXA3's healthy fold-change is exactly
0.000, so its 58.7× ratio is `log2(5.872 / 0.1)`, set by the constant rather than the data. It is the
only target with near-zero healthy signal, the other 19 having real signal on both sides.

### 4.5 An unplanned coherence check that passes

GATA6 is the most negative target at −2.24, or 0.21×. It is the classical-identity factor, and §9
independently established that PANC-1 sits below the panel mean on the classical programme, so a
classical-identity enhancer should be less acetylated in that line. Nobody designed this check. It fell
out of the per-locus table.

---

## 5. Effect size, reported as a range

| estimator | at published settings | span across all 12 settings |
|---|---|---|
| raw mean | +0.919 (1.89×) | +0.532 to +1.101 |
| **median** | **+0.852 (1.81×)** | +0.564 to +0.891 |
| **mean excluding zero denominator** | **+0.659 (1.58×)** | +0.457 to +0.676 |

The defensible statement is 1.5× to 1.8×, and the honest one adds that no single number is
well-supported. The existence of the effect is robust across every setting tested, while its magnitude
moves by roughly twofold under constants that no data constrains. Quoting 1.89× as the effect size
would misrepresent it.

---

## 6. Limitations, none of which the evidence removes

PANC-1 is an intermediate substrate. §9 showed it sits below the panel mean on both Moffitt
programmes, so it represents neither the basal nor the classical target set. This is a cell line
standing in for a tumour, and a poorly-typed one.

There is one healthy fold-change track per mark, against up to six averaged in the original p-value
run, so the healthy reference is noisier. This is the more conservative test and also the thinner one.

There are 20 targets, and the bootstrap interval [0.244, 1.679] reflects that honestly.

ATAC does not replicate, with a primary contrast p of 0.074. This is an H3K27ac-specific observation
rather than a general chromatin claim. Its matched controls were nominally significant and are
deliberately not used to rescue it, because significant subsets do not save a failed primary contrast.

It does not resurrect RAC. The targets come from a model whose essentiality claim is retracted (§15),
whose bistable framing is retired (§17), and which is unrescuable by gain (§18) or substrate (§19).
This says the gene set has a chromatin property. It says nothing about collapse predicting
essentiality.

---

## 7. What would confirm it

Four things would. A properly-typed substrate, meaning primary tumour or organoid H3K27ac rather than
an intermediate line. Multiple healthy references, to give the residual an error bar. A second mark
that replicates, since ATAC did not and whether this is acetylation-specific is unresolved. An
independent target set, because these 20 genes came from a retracted model.

Until then the claim stands as written in §1. Real, promoter-local, modest, and narrow.
