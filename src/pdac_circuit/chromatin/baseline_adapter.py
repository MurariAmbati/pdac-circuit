from __future__ import annotations

from dataclasses import asdict,dataclass,fields
import json
from pathlib import Path

@dataclass(frozen=True)
class EnformerAdapterConfig:
    bins: int = 896
    assay_features: int = 12
    state_features: int = 18
    perturbation_features: int = 22
    channels: int = 48
    layers: int = 6
    kernel_size: int = 7
    dilation_cycle: tuple[int,...] = (1,2,4,8,16,32)
    dropout: float = 0.10

    @property
    def condition_features(self) -> int:
        return self.assay_features + self.state_features + self.perturbation_features

    def validate(self) -> None:
        if self.bins < 8 or self.channels < 4 or self.layers < 1:
            raise ValueError("adapter bins, channels, and layers are too small")
        if self.kernel_size < 3 or self.kernel_size % 2 == 0:
            raise ValueError("adapter kernel_size must be odd and at least three")
        if not self.dilation_cycle or any(value < 1 for value in self.dilation_cycle):
            raise ValueError("adapter dilation cycle must be positive")
        if self.condition_features < 1:
            raise ValueError("adapter requires preregistered conditioning features")
        if not 0 <= self.dropout < 1:
            raise ValueError("adapter dropout must be in [0, 1)")

def load_adapter_config(path: str | Path) -> EnformerAdapterConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("schema") != "pdac-circuit.enformer-state-adapter/1":
        raise ValueError("invalid Enformer state-adapter config schema")
    values = dict(payload.get("model",{}))
    allowed = {field.name for field in fields(EnformerAdapterConfig)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown adapter config fields: {unknown}")
    if "dilation_cycle" in values:
        values["dilation_cycle"] = tuple(values["dilation_cycle"])
    config = EnformerAdapterConfig(**values)
    config.validate()
    return config

def EnformerStateAdapter(config: EnformerAdapterConfig | None = None):

    import torch
    import torch.nn as nn

    config = config or EnformerAdapterConfig()
    config.validate()

    class ResidualBlock(nn.Module):
        def __init__(self,dilation: int):
            super().__init__()
            padding = dilation * (config.kernel_size // 2)
            self.norm = nn.GroupNorm(8 if config.channels % 8 == 0 else 1,config.channels)
            self.depthwise = nn.Conv1d(
                config.channels,
                config.channels,
                config.kernel_size,
                padding=padding,
                dilation=dilation,
                groups=config.channels,
            )
            self.mix = nn.Conv1d(config.channels,config.channels,1)
            self.dropout = nn.Dropout(config.dropout)

        def forward(self,value):
            hidden = torch.nn.functional.silu(self.depthwise(self.norm(value)))
            return value + self.dropout(self.mix(hidden))

    class _Adapter(nn.Module):
        def __init__(self):
            super().__init__()
            self.config = config
            self.stem = nn.Conv1d(1,config.channels,9,padding=4)
            self.condition = nn.Sequential(
                nn.Linear(config.condition_features,config.channels),
                nn.SiLU(),
                nn.Linear(config.channels,config.channels * 2),
            )
            self.blocks = nn.ModuleList(
                ResidualBlock(config.dilation_cycle[index % len(config.dilation_cycle)])
                for index in range(config.layers)
            )
            self.residual = nn.Conv1d(config.channels,1,1)
            nn.init.zeros_(self.residual.weight)
            nn.init.zeros_(self.residual.bias)

        def forward(
            self,
            enformer_prediction,
            assay_features,
            state_features,
            perturbation_features,
        ):
            if enformer_prediction.ndim != 2 or enformer_prediction.shape[1] != config.bins:
                raise ValueError(
                    f"Enformer prediction must have shape (batch, {config.bins})"
                )
            condition = torch.cat(
                [assay_features,state_features,perturbation_features],dim=1
            )
            if condition.shape[1] != config.condition_features:
                raise ValueError(
                    f"adapter condition must have {config.condition_features} features"
                )
            log_profile = enformer_prediction.clamp_min(0).log1p()
            hidden = self.stem(log_profile[:,None,:])
            gamma,beta = self.condition(condition).chunk(2,dim=1)
            hidden = hidden * (1 + 0.1 * torch.tanh(gamma)[:,:,None]) + beta[:,:,None]
            for block in self.blocks:
                hidden = block(hidden)
            adapted_log = log_profile + self.residual(hidden).squeeze(1)
            return torch.expm1(adapted_log.clamp_min(0))

    return _Adapter()

def adapter_parameter_report(config: EnformerAdapterConfig | None = None) -> dict:
    model = EnformerStateAdapter(config)
    parameters = sum(value.numel() for value in model.parameters())
    return {
        "schema": "pdac-circuit.enformer-state-adapter-parameters/1",
        "config": asdict(config or EnformerAdapterConfig()),
        "parameters": parameters,
        "identity_initialized": True,
        "candidate_feature_access": False,
    }
