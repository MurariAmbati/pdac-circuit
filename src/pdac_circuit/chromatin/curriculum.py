from __future__ import annotations

TRAINING_STAGES = (
    "healthy_prior",
    "progression_state_residual",
    "signed_intervention_residual",
    "human_state_adaptation",
)

_STAGE_PREFIXES = {
    "healthy_prior": (
        "stem.",
        "to_model.",
        "blocks.",
        "mixers.",
        "assay_conditioner.",
        "baseline_head.",
        "uncertainty_head.",
    ),
    "progression_state_residual": (
        "stem.",
        "to_model.",
        "blocks.",
        "mixers.",
        "assay_conditioner.",
        "baseline_head.",
        "state_conditioner.",
        "state_basis_head.",
        "circuit_head.",
        "uncertainty_head.",
    ),
    "signed_intervention_residual": (
        "perturbation_conditioner.",
        "intervention_basis_head.",
        "intervention_head.",
        "uncertainty_head.",
    ),
}

def apply_training_stage(model, stage: str) -> dict:

    if stage not in TRAINING_STAGES:
        raise ValueError(f"unsupported training stage {stage!r}; choose from {TRAINING_STAGES}")
    prefixes = (
        None
        if getattr(getattr(model, "config", None), "architecture", "pdac_circuit")
        == "direct_conditional_cnn"
        else _STAGE_PREFIXES.get(stage)
    )
    trainable_names = []
    frozen_names = []
    for name, parameter in model.named_parameters():
        trainable = prefixes is None or name.startswith(prefixes)
        parameter.requires_grad_(trainable)
        (trainable_names if trainable else frozen_names).append(name)
    if not trainable_names:
        raise RuntimeError(f"training stage {stage!r} selected no parameters")
    trainable_parameters = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    total_parameters = sum(parameter.numel() for parameter in model.parameters())
    return {
        "schema": "pdac-circuit.chromatin-training-stage/1",
        "stage": stage,
        "trainable_tensors": len(trainable_names),
        "frozen_tensors": len(frozen_names),
        "trainable_parameters": trainable_parameters,
        "frozen_parameters": total_parameters - trainable_parameters,
        "total_parameters": total_parameters,
        "trainable_prefixes": list(prefixes) if prefixes is not None else ["*"],
    }
