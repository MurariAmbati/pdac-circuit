---
layout: default
title: Data
subtitle: Every corpus, its role, and how provenance is enforced.
permalink: /data/
---

All results rest on real public data. Nothing is simulated, imputed as a stand-in for measurement, or
generated to fill a gap; where data is unavailable the pipeline abstains rather than substitute a value.

## Corpora

<div class="tablewrap">
<table>
<thead><tr><th>Corpus</th><th>Contents</th><th>Role</th></tr></thead>
<tbody>
<tr><td>GRCh38 reference</td><td>Human genome assembly</td><td>Motif scanning, genome-wide off-target search, sequence windows</td></tr>
<tr><td>FANTOM5 CAGE</td><td>209,374 promoter peaks with TPM</td><td>Promoter-strength training; generator training set</td></tr>
<tr><td>ENCODE pancreas</td><td>ATAC-seq and H3K27ac peak sets and signal tracks</td><td>Enhancer-activity training; healthy chromatin reference</td></tr>
<tr><td>ENCODE PANC-1</td><td>Mark-matched PDAC cell-line panel on GRCh38</td><td>PDAC chromatin; cross-domain transfer tests</td></tr>
<tr><td>ENCODE fold-change</td><td>Fold-change-over-control bigWigs, PDAC and healthy</td><td>The H3K27ac residual analysis</td></tr>
<tr><td>Doench-2016</td><td>5,310 guides across 17 genes</td><td>gRNA on-target training and the held-out benchmark</td></tr>
<tr><td>Kim-2019</td><td>12,832 high-throughput SpCas9 guides</td><td>gRNA on-target training augmentation</td></tr>
<tr><td>Doench-2016 CFD matrix</td><td>Exact nucleotide-pair mismatch scores</td><td>Off-target specificity scoring</td></tr>
<tr><td>DepMap</td><td>CRISPR gene effect across 1,684 lines; expression</td><td>Essentiality readout, held out of every fit; co-expression graph</td></tr>
<tr><td>TCGA-PAAD</td><td>RSEM expression, GISTIC copy number, HM450 methylation, RPPA</td><td>Tumour multi-omics for target prioritisation</td></tr>
<tr><td>GTEx pancreas</td><td>Median TPM, normal tissue</td><td>Tumour-versus-normal expression contrast</td></tr>
<tr><td>CPTAC PDAC</td><td>238 tumours × 12,017 proteins</td><td>Protein-level evidence, including detection rate</td></tr>
<tr><td>TISCH2 / Peng-2019</td><td>57,443 single cells, 11,401 malignant</td><td>In-vivo malignant-cell regulatory graph</td></tr>
<tr><td>4DN PANC-1 Hi-C</td><td>A/B compartments, insulation, TAD boundaries</td><td>Three-dimensional genome context</td></tr>
<tr><td>GENCODE v46</td><td>Gene models</td><td>Transcription start sites</td></tr>
<tr><td>dbSNP common</td><td>Common human variants</td><td>Guide SNP-overlap flagging</td></tr>
<tr><td>Lambert TF catalogue, IntOGen, NCG</td><td>Transcription factors; cancer drivers</td><td>Target universe and driver annotation</td></tr>
</tbody>
</table>
</div>

## Provenance

Every corpus carries a manifest recording source URL, byte count, SHA-256, retrieval timestamp and a
`dataClass` of `REAL`. Three rules make this meaningful rather than decorative.

**Hashes are never fabricated.** A SHA-256 appears only when the bytes were actually hashed. Data behind an
authentication or data-use-agreement wall is recorded as a pointer and never bypassed; where a licence-walled
source would have been convenient — COSMIC, OncoKB — an open equivalent was used instead.

**Raw bytes stay out of version control.** The repository stores manifests, not corpora. This keeps history
small and makes the hash, rather than a copy, the thing that establishes identity.

**The hashes get exercised.** During this work the large reference corpora were cleared from disk to recover
space and then re-fetched. Because the manifests existed, the restored dbSNP and GRCh38 files could be checked
byte-for-byte against the recorded hashes and confirmed identical to what the results were computed on.
That is what provenance is for.

One honest gap is recorded rather than hidden: UCSC publishes no checksum for the mm9 assembly file
(`md5sum.txt` for that build lists only the per-chromosome tarballs), so the restored mm9 reference carries
`verified: false` in its manifest. Every other restored corpus verified against an upstream or
previously-recorded hash.

## Reproducibility

<div class="tablewrap">
<table>
<thead><tr><th>Mechanism</th><th>What it guarantees</th></tr></thead>
<tbody>
<tr><td>Manifest SHA-256</td><td>The data used is the data described</td></tr>
<tr><td><code>weight_sha256</code> in model manifests</td><td>A reported metric is bound to a specific checkpoint</td></tr>
<tr><td>Frozen predeploy fixtures</td><td>Reloading a checkpoint reproduces stored predictions to 1 × 10⁻⁴</td></tr>
<tr><td>Fixed seeds throughout</td><td>Retraining reproduces the reported figures</td></tr>
<tr><td>Pre-registered thresholds</td><td>A result clears a bar committed before training, or is reported as not clearing it</td></tr>
<tr><td>Leakage-controlled splits</td><td>Gene-grouped for guides; chromosome-held-out for regulatory parts</td></tr>
</tbody>
</table>
</div>

Superseded artifacts are marked in place rather than deleted. When the models were retrained on full data,
the earlier training records were annotated with a `SUPERSEDED` status, the reason, and a pointer to the
authoritative source — so an old number can still be read, but not mistaken for a current one.
