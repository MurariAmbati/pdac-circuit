from __future__ import annotations

from ..core.contract import OutputEnvelope
from ..data.genes import crispri_window

def select_repressor(target_tf: str, *, model=None, atac_index=None, snp_index=None, top_k: int = 3) -> OutputEnvelope:
    from ..grna.design import design_guides

    win=crispri_window(target_tf)
    if win is None:
        return OutputEnvelope.abstain(f"no TSS for {target_tf} in GENCODE; cannot place CRISPRi repressor", cert="abstain")
    env=design_guides([win], model=model, atac_index=atac_index, snp_index=snp_index, top_k=top_k)
    return env
