from __future__ import annotations

from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path
from typing import Any

@dataclass(frozen=True)
class ChromatinModelConfig:

    architecture: str = "pdac_circuit"
    sequence_length: int = 196_608
    bin_size: int = 128
    base_channels: int = 32
    d_model: int = 192
    n_layers: int = 8
    kernel_size: int = 7
    dilation_cycle: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
    landmark_tokens: int = 48
    landmark_routing: str = "dual_statistic"
    attention_heads: int = 6
    assay_features: int = 12
    state_features: int = 18
    progression_state_features: int = 4
    domain_state_features: int = 2
    perturbation_features: int = 22
    signed_perturbation_features: int = 19
    circuit_factors: int = 32
    dropout: float = 0.10
    gradient_checkpointing: bool = True

    @property
    def n_bins(self) -> int:
        return self.sequence_length // self.bin_size

    @property
    def downsample_stages(self) -> int:
        return int(math.log2(self.bin_size))

    def validate(self) -> None:
        if self.architecture not in {"pdac_circuit", "direct_conditional_cnn"}:
            raise ValueError("architecture must be pdac_circuit or direct_conditional_cnn")
        if self.sequence_length <= 0 or self.sequence_length % self.bin_size:
            raise ValueError("sequence_length must be positive and divisible by bin_size")
        if self.bin_size <= 0 or self.bin_size & (self.bin_size - 1):
            raise ValueError("bin_size must be a positive power of two")
        if self.d_model <= 0 or self.d_model % self.attention_heads:
            raise ValueError("d_model must be positive and divisible by attention_heads")
        if self.kernel_size < 3 or self.kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd and at least 3")
        if self.n_layers < 1 or self.landmark_tokens < 1:
            raise ValueError("n_layers and landmark_tokens must be positive")
        if self.architecture == "pdac_circuit":
            if self.landmark_routing not in {"dual_statistic", "mean_only"}:
                raise ValueError(
                    "pdac_circuit landmark_routing must be dual_statistic or mean_only"
                )
            if self.landmark_tokens < 2 or self.landmark_tokens % 2:
                raise ValueError(
                    "pdac_circuit landmark_tokens must be an even integer of at least two"
                )
            if self.landmark_tokens > self.n_bins:
                raise ValueError("pdac_circuit landmark_tokens cannot exceed output bins")
            if self.n_bins % (self.landmark_tokens // 2):
                raise ValueError(
                    "pdac_circuit output bins must be divisible by half the landmark tokens"
                )
        if not self.dilation_cycle or any(d < 1 for d in self.dilation_cycle):
            raise ValueError("dilation_cycle must contain positive integers")
        if min(self.assay_features, self.state_features, self.perturbation_features) < 0:
            raise ValueError("conditioning feature counts cannot be negative")
        if not 0 <= self.domain_state_features <= self.state_features:
            raise ValueError("domain_state_features must be within the state feature vector")
        if self.progression_state_features != 4:
            raise ValueError("PDAC progression_state_features is frozen at four states")
        if self.progression_state_features + self.domain_state_features > self.state_features:
            raise ValueError("progression and domain state features cannot overlap")
        if not 0 < self.signed_perturbation_features <= self.perturbation_features:
            raise ValueError(
                "signed_perturbation_features must be positive and no larger than "
                "perturbation_features"
            )
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")

@dataclass(frozen=True)
class ChromatinTrainConfig:

    micro_batch_size: int = 1
    gradient_accumulation: int = 32
    epochs: int = 40
    learning_rate: float = 2e-4
    weight_decay: float = 1e-3
    warmup_steps: int = 2_000
    max_steps: int = 200_000
    grad_clip: float = 1.0
    amp_dtype: str = "bfloat16"
    num_workers: int = 0
    prefetch_factor: int = 1
    checkpoint_every: int = 1_000
    eval_every: int = 1_000
    early_stopping_patience: int = 10
    minimum_validation_delta: float = 1e-4
    validation_max_batches: int | None = None
    loss_profile: float = 1.0
    loss_correlation: float = 0.25
    loss_uncertainty: float = 0.10
    loss_residual_delta: float = 0.50
    loss_perturbation_delta: float = 0.50
    loss_healthy_zero: float = 0.10
    loss_state_graph: float = 0.05
    loss_domain_invariance: float = 0.05
    seed: int = 20_260_620
    device: str = "auto"

    def validate(self) -> None:
        positive=(
            self.micro_batch_size,
            self.gradient_accumulation,
            self.epochs,
            self.max_steps,
            self.checkpoint_every,
            self.eval_every,
            self.early_stopping_patience,
        )
        if any(v < 1 for v in positive):
            raise ValueError("batch, accumulation, epoch, step, and interval values must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("learning_rate must be positive and weight_decay non-negative")
        if self.amp_dtype not in {"float16", "bfloat16", "float32"}:
            raise ValueError("amp_dtype must be float16, bfloat16, or float32")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if self.minimum_validation_delta < 0:
            raise ValueError("minimum_validation_delta cannot be negative")
        if self.validation_max_batches is not None and self.validation_max_batches < 1:
            raise ValueError("validation_max_batches must be positive when provided")
        loss_weights=(
            self.loss_profile,
            self.loss_correlation,
            self.loss_uncertainty,
            self.loss_residual_delta,
            self.loss_perturbation_delta,
            self.loss_healthy_zero,
            self.loss_state_graph,
            self.loss_domain_invariance,
        )
        if self.loss_profile <= 0 or any(weight < 0 for weight in loss_weights):
            raise ValueError("profile loss must be positive and all loss weights non-negative")

def _construct(cls, values: dict[str, Any]):
    allowed={f.name for f in fields(cls)}
    unknown=sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown {cls.__name__} fields: {unknown}")
    if cls is ChromatinModelConfig and "dilation_cycle" in values:
        values=dict(values)
        values["dilation_cycle"]=tuple(values["dilation_cycle"])
    obj=cls(**values)
    obj.validate()
    return obj

def load_chromatin_config(path: str | Path) -> tuple[ChromatinModelConfig, ChromatinTrainConfig, dict]:

    payload=json.loads(Path(path).read_text(encoding="utf-8"))
    model=_construct(ChromatinModelConfig, payload.get("model", {}))
    training=_construct(ChromatinTrainConfig, payload.get("training", {}))
    return model, training, payload

def config_payload(model: ChromatinModelConfig, training: ChromatinTrainConfig) -> dict[str, Any]:
    return {"model": asdict(model), "training": asdict(training)}
