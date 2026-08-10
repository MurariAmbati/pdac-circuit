from __future__ import annotations

REPRESS = "repress"
ACTIVATE = "activate"

BLOCK = "block"
QUARANTINE = "quarantine"
ALLOW = "allow"

GENE_ROLES: dict[str,dict] = {
    "TGIF1": {
        "role": "tumour_suppressor",
        "admissible": (ACTIVATE,),
        "status": BLOCK,
        "reason": ("TGIF1 loss accelerates KRAS-driven PDAC, increases metastatic propensity and "
                   "supports EMT/immune suppression; CRISPRi against TGIF1 is directionally "
                   "opposite to PDAC biology"),
    },
    "GATA6": {
        "role": "lineage_state_maintaining",
        "admissible": (ACTIVATE,),
        "status": QUARANTINE,
        "reason": ("GATA6 maintains classical epithelial identity; low GATA6 marks basal-like, "
                   "chemoresistant disease. Repression risks driving a classical->basal switch, "
                   "i.e. a more aggressive state. Retain as a state marker / positive control"),
    },
    "HNF4G": {
        "role": "stage_dependent",
        "admissible": (REPRESS,),
        "status": QUARANTINE,
        "reason": ("credible primary-tumour dependency, but loss can unmask FOXA1-driven "
                   "metastatic programmes; stage-dependent, not an unqualified repression target"),
    },
    "SMAD4": {
        "role": "tumour_suppressor",
        "admissible": (ACTIVATE,),
        "status": BLOCK,
        "reason": "canonical PDAC tumour suppressor, deleted in ~68% of tumours; repression is nonsensical",
    },
    "CDKN2A": {
        "role": "tumour_suppressor",
        "admissible": (ACTIVATE,),
        "status": BLOCK,
        "reason": "canonical PDAC tumour suppressor, deleted in ~63% of tumours",
    },
    "TP53": {
        "role": "tumour_suppressor",
        "admissible": (ACTIVATE,),
        "status": BLOCK,
        "reason": "canonical tumour suppressor; DepMap effect is positive (loss is tolerated/favoured)",
    },
    "BRCA2": {
        "role": "dna_repair_tumour_suppressor",
        "admissible": (),
        "status": QUARANTINE,
        "reason": ("DNA-repair tumour suppressor; repression is a genome-instability liability, not "
                   "a therapeutic direction. Relevant to synthetic lethality, not CRISPRi design"),
    },
    "KMT2C": {
        "role": "chromatin_tumour_suppressor",
        "admissible": (),
        "status": QUARANTINE,
        "reason": "COMPASS-family chromatin regulator with tumour-suppressive behaviour in PDAC",
    },
    "ARID1B": {
        "role": "chromatin_context_dependent",
        "admissible": (),
        "status": QUARANTINE,
        "reason": "SWI/SNF subunit; context-dependent, paralog-dependent direction",
    },
    "ATM": {
        "role": "dna_repair_tumour_suppressor",
        "admissible": (),
        "status": QUARANTINE,
        "reason": "DNA-damage-response tumour suppressor; repression increases instability",
    },
    "FBXW7": {
        "role": "tumour_suppressor",
        "admissible": (ACTIVATE,),
        "status": BLOCK,
        "reason": "ubiquitin-ligase tumour suppressor; loss stabilises oncoproteins including MYC",
    },
    "KLF5": {
        "role": "lineage_survival_oncogenic",
        "admissible": (REPRESS,),
        "status": ALLOW,
        "reason": ("lineage-survival TF, DepMap dependency in PDAC lines, controls subtype-spanning "
                   "highly interactive enhancers; repression is the supported direction"),
    },
    "MYC": {"role": "oncogene","admissible": (REPRESS,),"status": ALLOW,
            "reason": "canonical amplified oncogene and strong DepMap dependency"},
    "E2F1": {"role": "proliferation_oncogenic","admissible": (REPRESS,),"status": ALLOW,
             "reason": "proliferation TF, DepMap-essential and amplified"},
    "MYBL2": {"role": "proliferation_oncogenic","admissible": (REPRESS,),"status": ALLOW,
              "reason": "proliferation TF, DepMap-essential and amplified"},
    "SETDB1": {"role": "chromatin_oncogenic","admissible": (REPRESS,),"status": ALLOW,
               "reason": "H3K9 methyltransferase, DepMap-essential and amplified in PDAC"},
    "FOSL1": {"role": "ap1_oncogenic","admissible": (REPRESS,),"status": ALLOW,
              "reason": "AP-1 family, supports KRAS-driven programmes"},
    "TP63": {
        "role": "basal_state_driver",
        "admissible": (REPRESS,),
        "status": QUARANTINE,
        "reason": ("drives the basal/squamous programme; a repression target only in basal disease, "
                   "and its promoter is hypermethylated (silenced) in this classical-dominant cohort"),
    },
}

def classify(gene: str,direction: str = REPRESS,subtype: str | None = None) -> dict:
    entry = GENE_ROLES.get(gene.upper())
    if entry is None:
        return {
            "gene": gene,"direction": direction,"role": "unclassified","status": QUARANTINE,
            "admissible_directions": [],
            "reason": ("no curated role; direction not established. A driver label alone does not "
                       "imply that repression is the therapeutic direction"),
        }
    admissible = entry["admissible"]
    status = entry["status"]
    if status == ALLOW and direction not in admissible:
        status = BLOCK
    if status == BLOCK and direction in admissible:
        status = QUARANTINE
    out = {
        "gene": gene,"direction": direction,"role": entry["role"],"status": status,
        "admissible_directions": list(admissible),"reason": entry["reason"],
    }
    if subtype and gene.upper() == "TP63" and subtype == "basal" and direction == REPRESS:
        out["status"] = ALLOW
        out["reason"] = entry["reason"] + "; permitted here because the subtype axis is basal"
    if subtype and gene.upper() == "GATA6" and subtype == "basal":
        out["reason"] = entry["reason"] + "; GATA6 is already low in basal disease"
    return out

def gate_targets(rows: list[dict],subtype: str | None = None) -> list[dict]:
    kept = []
    for r in rows:
        direction = r.get("healthy_action") or REPRESS
        g = classify(r["gene"],direction,subtype)
        r = dict(r)
        r["intervention_gate"] = g
        if g["status"] == BLOCK:
            r["excluded"] = True
        kept.append(r)
    return kept

def summary(rows: list[dict]) -> dict:
    gated = [r for r in rows if "intervention_gate" in r]
    by = {}
    for r in gated:
        by.setdefault(r["intervention_gate"]["status"],[]).append(r["gene"])
    return {
        "n_gated": len(gated),
        "blocked": by.get(BLOCK,[]),
        "quarantined": by.get(QUARANTINE,[]),
        "allowed": by.get(ALLOW,[]),
        "note": ("a 'blocked' target is directionally contradicted by published PDAC genetics; a "
                 "'quarantined' target is state- or stage-dependent and must not be treated as an "
                 "unqualified repression candidate"),
    }
