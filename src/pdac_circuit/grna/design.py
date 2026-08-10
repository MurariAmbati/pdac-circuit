from __future__ import annotations

import json

import numpy as np

from ..core.contract import OutputEnvelope
from ..core.paths import MODELS, REGISTRY_JSON
from ..data.reference import fetch_sequence
from .datamodule import engineered_features
from .efficiency_model import GRNAModel
from .offtarget import score_guide_offtargets
from .scan import enumerate_protospacers
from .types import GuideCandidate

def predict_on_target(model: GRNAModel, contexts) -> np.ndarray:
    from ..harness.encoders import one_hot_batch
    from ..harness.trainer import Trainer, TrainConfig

    X=one_hot_batch(list(contexts), 30)
    cnn_pred=np.asarray(Trainer(TrainConfig(task="regression")).predict(model.cnn, X))
    gbm_pred=np.asarray(model.gbm.predict(engineered_features(list(contexts)))) if model.gbm is not None else cnn_pred
    ens=0.40 * cnn_pred + 0.60 * gbm_pred
    return model.to_unit(ens)

def _load_model() -> GRNAModel | None:
    p=MODELS / "grna_ontarget.pt"
    return GRNAModel.load(p) if p.exists() else None

def design_guides(
    loci: list[dict],
    *,
    model: GRNAModel | None = None,
    atac_index=None,
    snp_index=None,
    background_search=None,
    top_k: int = 10,
    seed: int = 20260620,
    genome_wide_offtarget: bool = True,
    offtarget_max_mm: int = 4,
) -> OutputEnvelope:
    pre=json.loads(REGISTRY_JSON.read_text(encoding="utf-8"))["prereg"]["module_V"]
    model=model or _load_model()
    if model is None:
        return OutputEnvelope.abstain("on-target model not trained; run `pdac train --only grna`", cert="abstain")

    cands: list[GuideCandidate]=[]
    for locus in loci:
        cands.extend(enumerate_protospacers(locus["chrom"], locus["start"], locus["end"]))
    if not cands:
        return OutputEnvelope.abstain("no NGG protospacers found in target loci", cert="abstain")

    on=predict_on_target(model, [c.context for c in cands])
    radius=float(json.loads((MODELS / "grna_ontarget.model.json").read_text())["metrics"].get("conformal_radius", 0.1)) if (MODELS / "grna_ontarget.model.json").exists() else 0.1
    for c, score in zip(cands, on):
        c.on_target=float(score)
        c.on_conf=(max(0.0, float(score) - radius), min(1.0, float(score) + radius))
        if atac_index is not None:
            c.in_open_chromatin=atac_index.any_overlap(c.chrom, c.start, c.end)

    search_seqs=background_search or []
    for locus in loci:
        s=fetch_sequence(locus["chrom"], max(0, locus["start"] - 5000), locus["end"] + 5000)
        search_seqs.append((locus["chrom"], s))
    cands.sort(key=lambda c: -(c.on_target or 0))
    for c in cands[: max(top_k * 4, 40)]:
        ot=score_guide_offtargets(c.protospacer, search_seqs, max_mm=pre["max_offtarget_mismatch"])
        c.cfd_specificity=ot["cfd_specificity"]
        c.mit_specificity=ot["mit_specificity"]
        c.off_targets=ot["off_targets"]

    scored=[c for c in cands if c.cfd_specificity is not None]
    if snp_index is not None:
        for c in scored:
            c.overlaps_common_snp=snp_index.any_overlap(c.chrom, c.start, c.end + 3)

    def rank_key(c: GuideCandidate):
        return ((c.on_target or 0) * (c.cfd_specificity or 0)
                * (1.1 if c.in_open_chromatin else 1.0)
                * (0.7 if c.overlaps_common_snp else 1.0))

    scored.sort(key=rank_key, reverse=True)

    shortlist=scored[: max(top_k * 3, 12)]
    for c in shortlist:
        c.cfd_specificity=None
        c.mit_specificity=None
        c.off_targets=[]
    scored=shortlist
    offtarget_scope="not_established_genome_wide_search_disabled"

    if genome_wide_offtarget:
        from .genome_offtarget import genome_wide_offtargets
        try:
            gw=genome_wide_offtargets([c.protospacer for c in shortlist], max_mm=offtarget_max_mm)
            for c in shortlist:
                r=gw[c.protospacer[:20].upper()]
                c.off_targets=r["off_targets"]
                c.cfd_specificity=r["cfd_specificity"]
                c.mit_specificity=r["mit_specificity"]
            offtarget_scope=f"genome_wide_hg38_le_{offtarget_max_mm}mm_both_strands"
            scored.sort(key=rank_key, reverse=True)
        except FileNotFoundError:
            offtarget_scope="not_established_hg38_absent"

    acceptable=[c for c in scored if (c.cfd_specificity or 0) >= pre["min_specificity"]]

    if not acceptable:
        best=scored[0] if scored else None
        detail=(f"best cfd_spec={best.cfd_specificity:.2f}"
                  if best is not None and best.cfd_specificity is not None
                  else "specificity not established")
        return OutputEnvelope.abstain(
            f"no guide clears specificity>={pre['min_specificity']} ({detail}; "
            f"off-target scope: {offtarget_scope})" if best else "no scored guides",
            cert="certified-negative",
        )
    top=acceptable[:top_k]
    payload={
        "guides": [c.to_dict() for c in top],
        "n_candidates": len(cands), "n_scored": len(scored), "n_acceptable": len(acceptable),
        "offtarget_scope": offtarget_scope,
        "offtarget_caveats": [
            "substitutions only; bulges are not modelled",
            "CFD is position-granular, not the exact Doench-2016 nucleotide-pair matrix",
            "counts are a lower bound on true off-target load",
        ],
    }
    return OutputEnvelope.ok(payload, cert="real")
