# RAC target dossiers

**Research Use Only.** Computational hypotheses, not validated dependencies. Every value
is real data; a layer with no measurement for a gene shows `–` rather than an imputed number.

Targets: **12** · subtype axis: **classical** · sealed studies touched: **False**

## Summary

| gene | verdict | ess. | sel. | CNA amp | β meth | protein (det.) | ATAC res | H3K27ac res | guide on-target |
|---|---|---|---|---|---|---|---|---|---|
| **BRCA2** | prioritise | 0.41 | -0.11 | 0.14 | – | +0.03 (38%) | – | 1.26 | – |
| **GATA6** | consider | 0.03 | 0.06 | 0.29 | – | +0.06 (67%) | – | -4.02 | – |
| **ZNF790** | consider | -0.03 | -0.04 | 0.22 | – | – | – | 1.65 | – |
| **SETDB1** | prioritise | 0.51 | -0.01 | 0.32 | – | +0.07 (100%) | – | 3.20 | – |
| **KMT2C** | consider | -0.12 | -0.11 | 0.23 | – | -0.11 (96%) | – | -1.04 | – |
| **E2F1** | prioritise | 0.41 | 0.12 | 0.23 | – | – | – | 2.76 | – |
| **SOX13** | consider | -0.18 | -0.07 | 0.31 | – | +0.12 (41%) | – | 0.81 | – |
| **AHR** | consider | -0.02 | 0.01 | 0.28 | – | – | – | 1.94 | – |
| **MYBL2** | prioritise | 0.60 | 0.06 | 0.26 | – | – | – | 3.05 | – |
| **AGR2** | deprioritise | 0.06 | 0.07 | 0.28 | – | -0.49 (100%) | – | 0.72 | – |
| **ZNF331** | consider | -0.23 | 0.01 | 0.17 | – | -0.11 (16%) | – | 3.35 | – |
| **SF3B1** | prioritise | 1.33 | -0.14 | 0.13 | – | +0.04 (100%) | – | 1.73 | – |

## Dossiers

### BRCA2 — prioritise

*DepMap-essential (0.42); strongly disease-up; PDAC-gained chromatin on one mark; IntOGen driver*

- **RAC**: convergence 0.694, collapse pct 0.870, motif-regulated disease genes 0, action **repress**, IntOGen driver
- **Is it real**: disease log2FC 6.05; DepMap essentiality 0.41 (PDAC-selectivity -0.11); CNA amp 0.14 / del 0.12; promoter β –; protein 0.03 detected 0.38
- **Is it active**: ATAC residual –, H3K27ac residual 1.26; Hi-C compartment A (eig 0.12), insulation 0.02, TAD boundary 90086 bp
- **Can I build it**: guide `–` (on-target –, off-risk –); promoter – strength –; enhancer –; simulated knock-down –, stable –

### GATA6 — consider

*strongly disease-up*

- **RAC**: convergence 0.679, collapse pct 0.899, motif-regulated disease genes 31, action **repress**, subtype **classical**
- **Is it real**: disease log2FC 5.92; DepMap essentiality 0.03 (PDAC-selectivity 0.06); CNA amp 0.29 / del 0.22; promoter β –; protein 0.06 detected 0.67
- **Is it active**: ATAC residual –, H3K27ac residual -4.02; Hi-C compartment A (eig 0.38), insulation 0.28, TAD boundary 14589 bp
- **Can I build it**: guide `–` (on-target –, off-risk –); promoter – strength –; enhancer –; simulated knock-down –, stable –

### ZNF790 — consider

*strongly disease-up; PDAC-gained chromatin on one mark*

- **RAC**: convergence 0.664, collapse pct 0.956, motif-regulated disease genes 0, action **repress**
- **Is it real**: disease log2FC 5.72; DepMap essentiality -0.03 (PDAC-selectivity -0.04); CNA amp 0.22 / del 0.05; promoter β –; protein – detected –
- **Is it active**: ATAC residual –, H3K27ac residual 1.65; Hi-C compartment A (eig 0.35), insulation -0.43, TAD boundary 84213 bp
- **Can I build it**: guide `–` (on-target –, off-risk –); promoter – strength –; enhancer –; simulated knock-down –, stable –

### SETDB1 — prioritise

*DepMap-essential (0.51); strongly disease-up; PDAC-gained chromatin on one mark; IntOGen driver*

- **RAC**: convergence 0.650, collapse pct 0.964, motif-regulated disease genes 0, action **repress**, IntOGen driver
- **Is it real**: disease log2FC 6.43; DepMap essentiality 0.51 (PDAC-selectivity -0.01); CNA amp 0.32 / del 0.03; promoter β –; protein 0.07 detected 1.00
- **Is it active**: ATAC residual –, H3K27ac residual 3.20; Hi-C compartment A (eig 1.50), insulation 0.96, TAD boundary 78737 bp
- **Can I build it**: guide `–` (on-target –, off-risk –); promoter – strength –; enhancer –; simulated knock-down –, stable –

### KMT2C — consider

*strongly disease-up; IntOGen driver*

- **RAC**: convergence 0.617, collapse pct 0.948, motif-regulated disease genes 0, action **repress**, IntOGen driver
- **Is it real**: disease log2FC 7.56; DepMap essentiality -0.12 (PDAC-selectivity -0.11); CNA amp 0.23 / del 0.08; promoter β –; protein -0.11 detected 0.96
- **Is it active**: ATAC residual –, H3K27ac residual -1.04; Hi-C compartment A (eig 0.33), insulation –, TAD boundary 121644 bp
- **Can I build it**: guide `–` (on-target –, off-risk –); promoter – strength –; enhancer –; simulated knock-down –, stable –

### E2F1 — prioritise

*DepMap-essential (0.41); strongly disease-up; PDAC-gained chromatin on one mark*

- **RAC**: convergence 0.617, collapse pct 1.000, motif-regulated disease genes 27, action **repress**
- **Is it real**: disease log2FC 6.85; DepMap essentiality 0.41 (PDAC-selectivity 0.12); CNA amp 0.23 / del 0.03; promoter β –; protein – detected –
- **Is it active**: ATAC residual –, H3K27ac residual 2.76; Hi-C compartment A (eig 0.83), insulation –, TAD boundary 58615 bp
- **Can I build it**: guide `–` (on-target –, off-risk –); promoter – strength –; enhancer –; simulated knock-down –, stable –

### SOX13 — consider

*strongly disease-up; PDAC-gained chromatin on one mark*

- **RAC**: convergence 0.596, collapse pct 0.730, motif-regulated disease genes 45, action **repress**
- **Is it real**: disease log2FC 6.88; DepMap essentiality -0.18 (PDAC-selectivity -0.07); CNA amp 0.31 / del 0.04; promoter β –; protein 0.12 detected 0.41
- **Is it active**: ATAC residual –, H3K27ac residual 0.81; Hi-C compartment A (eig 1.17), insulation 0.76, TAD boundary 38115 bp
- **Can I build it**: guide `–` (on-target –, off-risk –); promoter – strength –; enhancer –; simulated knock-down –, stable –

### AHR — consider

*strongly disease-up; PDAC-gained chromatin on one mark*

- **RAC**: convergence 0.595, collapse pct 0.771, motif-regulated disease genes 41, action **repress**
- **Is it real**: disease log2FC 9.57; DepMap essentiality -0.02 (PDAC-selectivity 0.01); CNA amp 0.28 / del 0.01; promoter β –; protein – detected –
- **Is it active**: ATAC residual –, H3K27ac residual 1.94; Hi-C compartment A (eig 0.36), insulation 0.08, TAD boundary 78641 bp
- **Can I build it**: guide `–` (on-target –, off-risk –); promoter – strength –; enhancer –; simulated knock-down –, stable –

### MYBL2 — prioritise

*DepMap-essential (0.60); strongly disease-up; PDAC-gained chromatin on one mark*

- **RAC**: convergence 0.592, collapse pct 0.919, motif-regulated disease genes 20, action **repress**
- **Is it real**: disease log2FC 5.60; DepMap essentiality 0.60 (PDAC-selectivity 0.06); CNA amp 0.26 / del 0.02; promoter β –; protein – detected –
- **Is it active**: ATAC residual –, H3K27ac residual 3.05; Hi-C compartment A (eig 0.20), insulation –, TAD boundary 232019 bp
- **Can I build it**: guide `–` (on-target –, off-risk –); promoter – strength –; enhancer –; simulated knock-down –, stable –

### AGR2 — deprioritise

*strongly disease-up; promoter hypermethylated (beta 0.59) -> already silenced; low protein abundance (-0.49); PDAC-gained chromatin on one mark*

- **RAC**: convergence 0.591, collapse pct 0.777, motif-regulated disease genes 0, action **repress**, subtype **classical**
- **Is it real**: disease log2FC 10.41; DepMap essentiality 0.06 (PDAC-selectivity 0.07); CNA amp 0.28 / del 0.01; promoter β –; protein -0.49 detected 1.00
- **Is it active**: ATAC residual –, H3K27ac residual 0.72; Hi-C compartment A (eig 0.12), insulation 0.29, TAD boundary 161567 bp
- **Can I build it**: guide `–` (on-target –, off-risk –); promoter – strength –; enhancer –; simulated knock-down –, stable –

### ZNF331 — consider

*strongly disease-up; protein barely detected (16%); PDAC-gained chromatin on one mark*

- **RAC**: convergence 0.584, collapse pct 0.917, motif-regulated disease genes 36, action **repress**
- **Is it real**: disease log2FC 6.52; DepMap essentiality -0.23 (PDAC-selectivity 0.01); CNA amp 0.17 / del 0.11; promoter β –; protein -0.11 detected 0.15
- **Is it active**: ATAC residual –, H3K27ac residual 3.35; Hi-C compartment A (eig 0.17), insulation -0.08, TAD boundary 25473 bp
- **Can I build it**: guide `–` (on-target –, off-risk –); promoter – strength –; enhancer –; simulated knock-down –, stable –

### SF3B1 — prioritise

*DepMap-essential (1.33); strongly disease-up; PDAC-gained chromatin on one mark; IntOGen driver*

- **RAC**: convergence 0.584, collapse pct 0.971, motif-regulated disease genes 0, action **repress**, IntOGen driver
- **Is it real**: disease log2FC 6.60; DepMap essentiality 1.33 (PDAC-selectivity -0.14); CNA amp 0.13 / del 0.03; promoter β –; protein 0.04 detected 1.00
- **Is it active**: ATAC residual –, H3K27ac residual 1.73; Hi-C compartment A (eig 0.10), insulation 0.46, TAD boundary 70079 bp
- **Can I build it**: guide `–` (on-target –, off-risk –); promoter – strength –; enhancer –; simulated knock-down –, stable –

