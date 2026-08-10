---
layout: default
title: "External audit response"
description: "Disposition of every finding from an external gap audit."
permalink: /reports/audit/
group: reports
order: 7
---

(≈200 findings across scientific, methodological, computational, validation, reproducibility, and
translational categories). This document records the honest disposition of every material finding:
**verified and fixed**, **already documented**, **correct and out of scope**, or **corrected**.

The audit is accurate and valuable. Its central verdict is one this project already reached and
documents throughout [REVIEW_RESPONSE.md]({{ '/reports/review/' | relative_url }}) and [COMPENDIUM.md]({{ '/reports/compendium/' | relative_url }}):

> a computational PDAC circuit-design prototype and methodological auditing framework — **not** a
> validated therapeutic platform, a source of cloning-ready constructs, a validated subtype sensor,
> a validated attractor system, or a source of safe final gRNAs.

Every checkable code-level claim in the audit was **verified against the code before acting** — the
same discipline this project applies to its own results — and every one checked was confirmed real.

---

## 1. Verified and FIXED in response to this audit (commit 21f6f9b)

| audit § | defect | verification | fix |
|---|---|---|---|
| **8.9** | Promoter scored on the wrong window: `promoter_window` returns 2500 bp (TSS−2000..+500), `score_promoters` truncates to `seq[:1000]` = TSS−2000..−1000, missing the core promoter | **Severe** — E2F1 scored **0.135** on the buggy far-upstream window vs **0.914** on the true core | score the TSS-centred 1000 bp (`up=500, down=500`) |
| **11.11** | Only `seqs[i][:200]` of the 1000 bp GAN promoter retained | confirmed in `_synthetic_promoter_library` | keep the full sequence |
| **11.12 / 14.2** | When a GAN promoter "wins", its sequence is never stored and `_optimize_promoter_seq` runs on the **native** promoter | confirmed at the selection branch | carry `prom["seq"]`; optimise the **selected** part |
| **11.13** | Immunogenicity scored on the native promoter even when GAN selected | confirmed (`immunogenicity_risk(prom_seq[:300])`) | score the selected part |
| **Blocker 5 / 14.10** | `acceptable=True` hardcoded despite sequence-optimisation failures (20/20 saved circuits GC-violated, 3/20 kept BsmBI) | confirmed in the saved runs and the `CircuitScore` line | `acceptable = not seqopt["unsatisfied"]` (fail-closed) |
| **15.18 / 3.5** | Stale `run_basal`/`run_classical`/`gated_constructs` carried pre-repair `cfd_specificity=1.0` / `off_risk=0.0` / `robustness=1.0` with no superseded marker | confirmed | added a `SUPERSEDED` status header with reason to each |

**Consequence — and it is the correct one.** With the §14 off-target repair now honest,
`select_repressor` finds no guide that clears genome-wide specificity, so the pipeline **ABSTAINS
(0 circuits)** rather than emitting circuits with fabricated `off_risk = 0.0`. That is precisely the
audit's Blocker 1 ("none of the current guide sequences should advance"), now enforced by the code.
`results/run_classical_fixed.json` records the honest certified-negative.

---

## 2. Verified and ALREADY DOCUMENTED (the review arc §1–§28 reached these independently)

The audit's major scientific findings were already this project's own conclusions:

| audit finding | where already recorded |
|---|---|
| RAC did not demonstrate bistability; behaves like degree; adds nothing beyond covariates; correctly retracted (Blocker 7, §16) | §1, §15, §15b, §17, §18; ADDENDUM_DYNAMICS |
| All four final guides fail the corrected specificity threshold (Blocker 1, §10.8) | §14, §24 |
| The robustness metric is invalid — a dead circuit scores 1.0 (Blocker 4, §13.10–13.11) | §11 |
| Subtype-specificity did not survive validation; PANC-1 is not a clean subtype model (Blocker 6, §5.13) | §9 |
| The original off-target search covered ~0.0013% of the genome (§10.7) | §13 |
| GAN validation is circular and did not beat real promoters; strength scale uninformative (§11.6–11.7) | §12 |
| Cross-cohort TCGA-vs-GTEx, not matched tumour-normal (§4.1–4.2) | §2, ADDENDUM_CHROMATIN §2 |
| MCDA weights zeroed tumour-normal specificity and oncogenic confidence (§5.4) | §10, memory |
| Module I permutation was selection-inflated; corrected to p≈0.013 (§5.5) | §10 |
| H3K27ac is the one surviving result, PANC-1-only, ~1.5–1.8×, ATAC does not replicate (§17) | §25–§28, ADDENDUM_CHROMATIN |
| "REAL" means real input data, not experimentally validated (§3.6) | COMPENDIUM §0, RUO banners |
| The "random forest" is actually HistGradientBoosting (§8.3) | module_metrics_scorecard.json |

The audit independently reproducing these conclusions is corroboration that the record is honest.

---

## 3. Correct, and genuinely OUT OF SCOPE (require experiments or new data, not code)

The audit is right that these are missing; they are not defects to patch but the boundary between an
in-silico prototype and validated synthetic biology. They are already listed as limitations in
COMPENDIUM §9 and ADDENDUM_CHROMATIN §6–§7.

- **All of §19** — no wet-lab: no cloning, reporter assay, CRISPRi knockdown measurement, truth-table
  measurement, organoid/in-vivo study, experimental off-target assay (GUIDE-/CIRCLE-/CHANGE-seq).
- **§22 Priority 2/3** — the minimum wet-lab package and translational validation.
- **Biological-substrate limits** (§4.3 cell-composition confound, §4.5 bulk vs ductal chromatin,
  §4.6 single cell line, §17.2–17.4) — need primary tumour/organoid data and matched normals.
- **Physical-construct completeness** (Blocker 2, §7.2, §12.15, §14.11–14.12) — no delivery vector,
  effector CDS, cassettes, or GenBank output. The circuits are parameterised hypotheses.
- **Kinetic calibration** (§12.6–12.7, §13.4) — ODE parameters are engineering priors, not fitted.

These are correctly framed by the audit as the decisive next steps, and this repository does not
claim to have crossed them.

---

## 4. Design limitations acknowledged (correct; not a quick fix)

- **§12.1 `_build_circuit(tf)` ignores its target** — verified: identical three-node topology and
  kinetics for every target; the target name changes the scores, not the ODE system. A genuine fix
  requires target-specific, experimentally grounded kinetics (§3 out of scope), so this is recorded
  as a limitation rather than patched with more arbitrary mappings.
- **§8.6 promoter output is a percentile, not physical strength**, and **§8.7 conformal-interval unit
  mismatch** — correct; the `[0,1]` "strength" is a training-CDF rank. Left as-is with this
  clarification because relabelling touches many artifacts; the interpretation is now stated here.
- **§15.16 deep-pipeline composite = 0.0** — the deep path does not run the full score-finalisation;
  ordering there is Pareto-mechanics only. The regular pipeline (now abstaining) is the honest one.

---

## 5. Minor corrections to the audit

- **§8.3** calls the non-CNN learner a "random forest"; it is `HistGradientBoostingRegressor`. The
  audit correctly notes the naming should change — this was already recorded in the module scorecard.
- The audit could not access `data/raw/` or model weights (stated up front). Those exist locally and
  are sha256-recorded in `data/manifests/`; the archive delivered to the auditor was code+results
  only, which is why several analyses "cannot be rerun from this archive alone" (§3.1–3.2). This is a
  packaging limitation, not missing data.

---

## 6. Net effect

The audit changed nothing about the project's honest standing and improved its engineering integrity:
six concrete defects fixed, three stale artifacts marked superseded, and the pipeline now **abstains
correctly** instead of emitting circuits built on invalidated off-target numbers. The single
surviving biological result (H3K27ac, §25–§28) is untouched by any of this — it never depended on the
circuit-assembly path. The correct one-line description is unchanged:

> a computational prototype and auditing framework that has not yet produced a complete, safe,
> experimentally validated synthetic circuit; the decisive next step is wet-lab validation, not more
> simulated designs.
