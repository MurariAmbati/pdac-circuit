# Direction-aware target ranking

**Research Use Only.** Computational hypotheses. The direction is decided from gene role
*before* scoring, and evidence is signed for that direction. Attractor-collapse is shown as a
descriptor only and carries **zero weight**: its discrimination did not beat network degree
and the essentiality claim is retracted (see REVIEW_RESPONSE.md).

Of **20** input targets: **4** rankable, **4** quarantined, **12** unclassified.

## Rankable candidates

| gene | modality | score | role | disease log2FC | CNA amp | beta | protein det. |
|---|---|---|---|---|---|---|---|
| **SETDB1** | CRISPRi | 0.778 | chromatin_oncogenic | – | 0.32 | 0.03 | 1.00 |
| **MYBL2** | CRISPRi | 0.714 | proliferation_oncogenic | – | 0.26 | 0.21 | – |
| **E2F1** | CRISPRi | 0.692 | proliferation_oncogenic | – | 0.23 | 0.02 | – |
| **FOSL1** | CRISPRi | 0.418 | ap1_oncogenic | – | 0.08 | 0.28 | 0.15 |

## Quarantined — state- or stage-dependent, not therapeutic candidates

- **BRCA2** (dna_repair_tumour_suppressor): DNA-repair tumour suppressor; repression is a genome-instability liability, not a therapeutic direction. Relevant to synthetic lethality, not CRISPRi design
- **GATA6** (lineage_state_maintaining): GATA6 maintains classical epithelial identity; low GATA6 marks basal-like, chemoresistant disease. Repression risks driving a classical->basal switch, i.e. a more aggressive state. Retain as a state marker / positive control
- **KMT2C** (chromatin_tumour_suppressor): COMPASS-family chromatin regulator with tumour-suppressive behaviour in PDAC
- **ATM** (dna_repair_tumour_suppressor): DNA-damage-response tumour suppressor; repression increases instability

## Unclassified — direction not established

`ZNF790`, `SOX13`, `AHR`, `AGR2`, `ZNF331`, `SF3B1`, `SMAD3`, `HOXA3`, `ZNF528`, `FAM83A`, `ZNF93`, `ZNF85`

A driver label alone does not imply repression is the therapeutic direction. These are not
ranked, and are not safe to treat as CRISPRi candidates without a curated role.

