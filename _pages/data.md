---
layout: default
title: Data
subtitle: Every corpus, the role it plays, and how provenance is enforced rather than asserted.
permalink: /data/
---

Every result rests on real public data. Nothing is simulated, and nothing is imputed as a stand-in for a
measurement that was not made. Where data is unavailable the pipeline abstains rather than substituting a
value, which is why several analyses in this project terminate in a certified negative instead of a number.

## Corpora

<div class="tablewrap">
<table>
<thead><tr><th>Corpus</th><th>Contents</th><th>Role</th></tr></thead>
<tbody>
<tr><td>GRCh38 reference</td><td>Human genome assembly</td><td>Motif scanning, genome-wide off-target search, sequence-window extraction</td></tr>
<tr><td>FANTOM5 CAGE</td><td>209,374 promoter peaks with TPM</td><td>Promoter-strength training and the generator training set</td></tr>
<tr><td>ENCODE pancreas</td><td>ATAC-seq and H3K27ac peak sets, signal tracks, 54 ChIP BAMs</td><td>Enhancer-activity training and the healthy chromatin reference</td></tr>
<tr><td>ENCODE PANC-1</td><td>Mark-matched PDAC cell-line panel on GRCh38</td><td>PDAC chromatin and the cross-domain transfer tests</td></tr>
<tr><td>ENCODE fold-change</td><td>Fold-change-over-control bigWigs for PDAC and healthy</td><td>The H3K27ac residual analysis</td></tr>
<tr><td>Doench-2016</td><td>5,310 guides across 17 genes</td><td>gRNA on-target training and the fixed held-out benchmark</td></tr>
<tr><td>Kim-2019</td><td>12,832 high-throughput SpCas9 guides</td><td>gRNA on-target training augmentation</td></tr>
<tr><td>Doench-2016 CFD matrix</td><td>Exact nucleotide-pair mismatch scores</td><td>Off-target specificity scoring</td></tr>
<tr><td>DepMap</td><td>CRISPR gene effect across 1,684 lines, plus expression</td><td>Essentiality readout held out of every fit, and the co-expression graph</td></tr>
<tr><td>TCGA-PAAD</td><td>RSEM expression, GISTIC copy number, HM450 methylation, RPPA</td><td>Tumour multi-omics for target prioritisation</td></tr>
<tr><td>GTEx pancreas</td><td>Median TPM in normal tissue</td><td>The tumour against normal expression contrast</td></tr>
<tr><td>CPTAC PDAC</td><td>238 tumours by 12,017 proteins</td><td>Protein-level evidence including per-protein detection rate</td></tr>
<tr><td>TISCH2 and Peng-2019</td><td>57,443 single cells, of which 11,401 malignant</td><td>The in-vivo malignant-cell regulatory graph</td></tr>
<tr><td>4DN PANC-1 Hi-C</td><td>A and B compartments, insulation, TAD boundaries</td><td>Three-dimensional genome context</td></tr>
<tr><td>GENCODE v46</td><td>Gene models</td><td>Transcription start site coordinates</td></tr>
<tr><td>dbSNP common</td><td>Common human variants</td><td>Flagging guides that overlap common variation</td></tr>
<tr><td>Lambert catalogue, IntOGen, NCG</td><td>Transcription factors and cancer drivers</td><td>The target universe and its driver annotation</td></tr>
</tbody>
</table>
</div>

The single-cell layer is worth a note, because it produced a result that runs against intuition. The
malignant-cell graph built by `scripts/build_malignant_graph.py` from 11,401 malignant cells is in principle
a more faithful in-vivo substrate than a panel of cultured cell lines. It nonetheless performs worse for
predicting CRISPR essentiality, at AUC around 0.51 to 0.55 against roughly 0.65 for the cell-line graph. The
explanation is that the readout being predicted was itself measured in cultured cell lines, so the cell-line
substrate matches the measurement rather than the biology. That is a caution about what a validation target
actually represents, and it is the kind of thing that is easy to get backwards.

## Provenance

Every corpus carries a manifest under `data/manifests/` recording its source URL, byte count, SHA-256,
retrieval timestamp and a data class of REAL. Three rules keep that from being decorative.

Hashes are never fabricated. A SHA-256 appears only where the bytes were actually hashed, and data behind an
authentication or data-use-agreement wall is recorded as a pointer and never bypassed. Where a licence-walled
source would have been convenient, COSMIC and OncoKB being the obvious cases, an open equivalent was used
instead rather than working around the licence.

Raw bytes stay out of version control. The repository stores manifests rather than corpora, which keeps
history small and makes the hash rather than a copy the thing that establishes identity.

The hashes are exercised rather than merely written. During this work the large reference corpora were
cleared from disk to recover space and later re-fetched. Because the manifests existed, the restored dbSNP
and GRCh38 files could be checked byte for byte against the recorded hashes and confirmed identical to the
files the results had been computed on. That is what provenance is for, and it is the only reason the
restoration could be treated as a restoration rather than as a substitution.

One gap is recorded rather than hidden. UCSC publishes no checksum for the mm9 assembly file, since the
`md5sum.txt` for that build lists only the per-chromosome tarballs, so the restored mm9 reference carries
`verified: false` in its manifest. Every other restored corpus verified against either an upstream published
hash or a previously recorded one.

## Reproducibility

<div class="tablewrap">
<table>
<thead><tr><th>Mechanism</th><th>What it guarantees</th></tr></thead>
<tbody>
<tr><td>Manifest SHA-256</td><td>The data used is the data described</td></tr>
<tr><td><code>weight_sha256</code> in model manifests</td><td>A reported metric is bound to one specific checkpoint</td></tr>
<tr><td>Frozen predeploy fixtures</td><td>Reloading a checkpoint reproduces stored predictions to within 1e-4</td></tr>
<tr><td>Fixed seeds throughout</td><td>Retraining reproduces the reported figures</td></tr>
<tr><td>Pre-registered thresholds</td><td>A result clears a bar committed before training, or is reported as not clearing it</td></tr>
<tr><td>Leakage-controlled splits</td><td>Gene-grouped for guides, chromosome-held-out for regulatory parts</td></tr>
</tbody>
</table>
</div>

Superseded artifacts are marked in place rather than deleted. When the four models were retrained on full
data, the earlier training records were annotated with a SUPERSEDED status, the reason for the supersession,
and a pointer to the authoritative source. An old number can therefore still be read in its original context,
but it cannot be mistaken for a current one. The same convention was applied to the pipeline runs that
predate the off-target repair, since those carry the pre-repair specificity values and would otherwise look
like ordinary results.
