from __future__ import annotations

from dataclasses import dataclass

from .config import ChromatinModelConfig

@dataclass
class ChromatinOutput:
    mean: object
    baseline: object
    state_residual: object
    perturbation_residual: object
    residual: object
    log_variance: object
    circuit_factors: object
    intervention_factors: object
    intervention_axis_potentials: object | None=None
    counterfactual_factors: object | None=None
    domain_counterfactual_factors: object | None=None

def reverse_complement_one_hot(sequence):

    return sequence[:, [3, 2, 1, 0], :].flip(-1)

def _group_count(channels: int) -> int:
    for candidate in (16, 8, 4, 2):
        if channels % candidate == 0:
            return candidate
    return 1

def _torch_modules():
    import torch
    import torch.nn as nn

    return torch, nn

class PDACircuitFormer:

    def __new__(cls, config: ChromatinModelConfig | None = None):
        config=config or ChromatinModelConfig()
        config.validate()
        torch, nn=_torch_modules()

        class DownsampleBlock(nn.Module):
            def __init__(self, in_channels: int, out_channels: int):
                super().__init__()
                self.block=nn.Sequential(
                    nn.Conv1d(in_channels, out_channels, 9, stride=2, padding=4, bias=False),
                    nn.GroupNorm(_group_count(out_channels), out_channels),
                    nn.GELU(),
                    nn.Conv1d(
                        out_channels,
                        out_channels,
                        5,
                        padding=2,
                        groups=out_channels,
                        bias=False,
                    ),
                    nn.GroupNorm(_group_count(out_channels), out_channels),
                    nn.GELU(),
                )

            def forward(self, x):
                return self.block(x)

        class FeatureConditioner(nn.Module):
            def __init__(self, n_features: int, channels: int):
                super().__init__()
                self.n_features=n_features
                if n_features:
                    self.net=nn.Sequential(
                        nn.Linear(n_features, channels),
                        nn.SiLU(),
                        nn.Linear(channels, channels * 2),
                    )
                else:
                    self.net=None

            def forward(self, x, features):
                if self.net is None:
                    return x
                if features is None:
                    raise ValueError(f"expected {self.n_features} conditioning features")
                if features.ndim != 2 or features.shape[1] != self.n_features:
                    raise ValueError(
                        f"conditioning features must have shape (batch, {self.n_features})"
                    )
                gamma, beta=self.net(features).chunk(2, dim=-1)
                gamma=0.1 * torch.tanh(gamma).unsqueeze(-1)
                beta=beta.unsqueeze(-1)
                return x * (1.0 + gamma) + beta

        class GatedDilatedBlock(nn.Module):
            def __init__(self, channels: int, dilation: int):
                super().__init__()
                padding=dilation * (config.kernel_size // 2)
                self.norm=nn.GroupNorm(_group_count(channels), channels)
                self.depthwise=nn.Conv1d(
                    channels,
                    channels,
                    config.kernel_size,
                    padding=padding,
                    dilation=dilation,
                    groups=channels,
                )
                self.pointwise=nn.Conv1d(channels, channels * 2, 1)
                self.project=nn.Conv1d(channels, channels, 1)
                self.dropout=nn.Dropout(config.dropout)

            def forward(self, x):
                h=self.depthwise(self.norm(x))
                value, gate=self.pointwise(h).chunk(2, dim=1)
                h=value * torch.sigmoid(gate)
                return x + self.dropout(self.project(h))

        class LandmarkMixer(nn.Module):
            def __init__(self, channels: int):
                super().__init__()
                self.query_norm=nn.LayerNorm(channels)
                self.landmark_norm=nn.LayerNorm(channels)
                self.attn=nn.MultiheadAttention(
                    channels,
                    config.attention_heads,
                    dropout=config.dropout,
                    batch_first=True,
                )
                self.out_norm=nn.LayerNorm(channels)
                self.ffn=nn.Sequential(
                    nn.Linear(channels, channels * 2),
                    nn.GELU(),
                    nn.Dropout(config.dropout),
                    nn.Linear(channels * 2, channels),
                    nn.Dropout(config.dropout),
                )

            def _content_routed_landmarks(self, x):

                n_bins=x.shape[-1]
                n_landmarks=min(config.landmark_tokens, n_bins)
                regions=n_landmarks // 2
                if regions < 1 or n_bins % regions:
                    raise ValueError(
                        "PDAC landmark routing requires bins divisible by half the landmarks"
                    )
                span=n_bins // regions
                grouped=x.reshape(x.shape[0], x.shape[1], regions, span)
                mean_context=grouped.mean(dim=-1).transpose(1, 2)
                if config.landmark_routing == "mean_only":
                    return torch.stack((mean_context, mean_context), dim=2).reshape(
                        x.shape[0], regions * 2, x.shape[1]
                    )
                energy=grouped.square().mean(dim=1)
                centered=energy - energy.mean(dim=-1, keepdim=True)
                standardized=centered / centered.square().mean(
                    dim=-1, keepdim=True
                ).add(1e-6).sqrt()
                weights=torch.softmax(standardized.clamp(-8.0, 8.0), dim=-1)
                content=torch.einsum("bcrs,brs->brc", grouped, weights)
                return torch.stack((mean_context, content), dim=2).reshape(
                    x.shape[0], regions * 2, x.shape[1]
                )

            def _relative_landmark_bias(self, n_bins, regions, *, device, dtype):

                query_position=(
                    torch.arange(n_bins, device=device, dtype=torch.float32) + 0.5
                ) / n_bins
                region_position=(
                    torch.arange(regions, device=device, dtype=torch.float32) + 0.5
                ) / regions
                landmark_position=region_position.repeat_interleave(2)
                signed_distance=landmark_position.unsqueeze(0) - query_position.unsqueeze(1)
                head_fraction=torch.linspace(
                    0.0,
                    1.0,
                    config.attention_heads,
                    device=device,
                    dtype=torch.float32,
                )
                scales=torch.pow(1.0 / regions, 1.0 - head_fraction).clamp_min(1e-4)
                direction=(
                    torch.arange(
                        config.attention_heads,
                        device=device,
                        dtype=torch.float32,
                    ).remainder(3)
                    - 1.0
                )
                normalized=signed_distance.unsqueeze(0) / scales[:, None, None]
                bias=-normalized.abs() + 0.25 * direction[:, None, None] * torch.tanh(
                    normalized
                )
                return bias.to(dtype=dtype)

            def forward(self, x):
                tokens=x.transpose(1, 2)
                landmarks=self._content_routed_landmarks(x)
                regions=landmarks.shape[1] // 2
                attention_bias=self._relative_landmark_bias(
                    tokens.shape[1],
                    regions,
                    device=tokens.device,
                    dtype=tokens.dtype,
                ).repeat(tokens.shape[0], 1, 1)
                mixed, _=self.attn(
                    self.query_norm(tokens),
                    self.landmark_norm(landmarks),
                    self.landmark_norm(landmarks),
                    attn_mask=attention_bias,
                    need_weights=False,
                )
                tokens=tokens + mixed
                tokens=tokens + self.ffn(self.out_norm(tokens))
                return tokens.transpose(1, 2)

        class _PDACircuitFormer(nn.Module):
            def __init__(self):
                super().__init__()
                self.config=config
                channels=[]
                for stage in range(config.downsample_stages):
                    channels.append(min(config.d_model, config.base_channels * (2 ** (stage // 2))))
                stem=[]
                in_channels=4
                for out_channels in channels:
                    stem.append(DownsampleBlock(in_channels, out_channels))
                    in_channels=out_channels
                self.stem=nn.ModuleList(stem)
                self.to_model=nn.Conv1d(in_channels, config.d_model, 1)
                self.blocks=nn.ModuleList(
                    GatedDilatedBlock(
                        config.d_model,
                        config.dilation_cycle[i % len(config.dilation_cycle)],
                    )
                    for i in range(config.n_layers)
                )
                self.mixers=nn.ModuleDict(
                    {
                        str(i): LandmarkMixer(config.d_model)
                        for i in sorted({config.n_layers // 3, (2 * config.n_layers) // 3})
                        if 0 <= i < config.n_layers
                    }
                )
                self.assay_conditioner=FeatureConditioner(
                    config.assay_features + config.domain_state_features,
                    config.d_model,
                )
                state_features=config.assay_features + config.state_features
                perturbation_features=state_features + config.perturbation_features
                self.state_conditioner=FeatureConditioner(state_features, config.d_model)
                self.perturbation_conditioner=FeatureConditioner(
                    perturbation_features, config.d_model
                )
                self.baseline_head=nn.Sequential(
                    nn.GroupNorm(_group_count(config.d_model), config.d_model),
                    nn.GELU(),
                    nn.Conv1d(config.d_model, 1, 1),
                )
                self.state_basis_head=nn.Sequential(
                    nn.GroupNorm(_group_count(config.d_model), config.d_model),
                    nn.GELU(),
                    nn.Conv1d(config.d_model, config.circuit_factors, 1),
                )
                self.intervention_basis_head=nn.Sequential(
                    nn.GroupNorm(_group_count(config.d_model), config.d_model),
                    nn.GELU(),
                    nn.Conv1d(config.d_model, config.circuit_factors, 1),
                )
                self.uncertainty_head=nn.Conv1d(config.d_model, 1, 1)
                self.circuit_head=nn.Sequential(
                    nn.LayerNorm(config.d_model),
                    nn.Linear(config.d_model, config.circuit_factors),
                    nn.Tanh(),
                )
                self.intervention_head=nn.Sequential(
                    nn.LayerNorm(config.d_model),
                    nn.Linear(
                        config.d_model,
                        config.signed_perturbation_features * config.circuit_factors,
                    ),
                    nn.Tanh(),
                )
                nn.init.zeros_(self.circuit_head[1].weight)
                nn.init.zeros_(self.circuit_head[1].bias)
                nn.init.zeros_(self.intervention_head[1].weight)
                nn.init.zeros_(self.intervention_head[1].bias)

            def _run_block(self, block, x):
                if self.config.gradient_checkpointing and self.training and x.requires_grad:
                    from torch.utils.checkpoint import checkpoint

                    return checkpoint(block, x, use_reentrant=False)
                return block(x)

            def encode(self, sequence):
                if sequence.ndim != 3 or sequence.shape[1] != 4:
                    raise ValueError("sequence must have shape (batch, 4, length)")
                if sequence.shape[-1] != self.config.sequence_length:
                    raise ValueError(
                        f"expected sequence length {self.config.sequence_length}, got {sequence.shape[-1]}"
                    )
                h=sequence
                for block in self.stem:
                    h=self._run_block(block, h)
                h=self.to_model(h)
                for i, block in enumerate(self.blocks):
                    h=self._run_block(block, h)
                    mixer=self.mixers[str(i)] if str(i) in self.mixers else None
                    if mixer is not None:
                        h=self._run_block(mixer, h)
                return h

            @staticmethod
            def _condition_concat(reference, *parts):
                arrays=[]
                for part in parts:
                    if part is not None:
                        arrays.append(part)
                if not arrays:
                    return reference.new_zeros((reference.shape[0], 0))
                return torch.cat(arrays, dim=1)

            def _state_representation(self, h, assay_features, state_features):
                state_condition=self._condition_concat(
                    h,
                    assay_features,
                    state_features,
                )
                state_h=self.state_conditioner(h, state_condition)
                factors=self.circuit_head(state_h.mean(dim=-1))
                return state_h, factors

            def _counterfactual_progression_factors(
                self, h, assay_features, state_features
            ):
                factors=[]
                n_states=self.config.progression_state_features
                for state_index in range(n_states):
                    counterfactual=state_features.clone()
                    counterfactual[:, :n_states]=0.0
                    counterfactual[:, state_index]=1.0
                    _, state_factors=self._state_representation(
                        h, assay_features, counterfactual
                    )
                    factors.append(state_factors)
                return torch.stack(factors, dim=1)

            def _domain_counterfactual_factors(self, h, assay_features, state_features):
                if self.config.domain_state_features != 2:
                    return None
                factors=[]
                for domain_index in range(2):
                    counterfactual=state_features.clone()
                    counterfactual[:, -2:]=0.0
                    counterfactual[:, -2 + domain_index]=1.0
                    _, state_factors=self._state_representation(
                        h, assay_features, counterfactual
                    )
                    factors.append(state_factors)
                return torch.stack(factors, dim=1)

            def forward(
                self,
                sequence,
                assay_features,
                state_features,
                perturbation_features=None,
                disease_mask=None,
                ablate_state_residual=False,
                ablate_intervention_residual=False,
            ):
                h=self.encode(sequence)
                domain_features=(
                    state_features[:, -self.config.domain_state_features :]
                    if self.config.domain_state_features
                    else None
                )
                baseline_condition=self._condition_concat(
                    h, assay_features, domain_features
                )
                baseline_h=self.assay_conditioner(h, baseline_condition)
                baseline_raw=self.baseline_head(baseline_h).squeeze(1)

                if perturbation_features is None and self.config.perturbation_features:
                    perturbation_features=h.new_zeros(
                        (h.shape[0], self.config.perturbation_features)
                    )
                state_h, circuit_factors=self._state_representation(
                    h, assay_features, state_features
                )
                state_basis=self.state_basis_head(baseline_h)
                state_residual=torch.einsum(
                    "bf,bfn->bn", circuit_factors, state_basis
                ) / (self.config.circuit_factors ** 0.5)
                if disease_mask is not None:
                    if disease_mask.ndim == 1:
                        disease_mask=disease_mask[:, None]
                    state_residual=state_residual * disease_mask.to(state_residual.dtype)
                if ablate_state_residual:
                    state_residual=state_residual * 0.0

                perturbation_condition=self._condition_concat(
                    h,
                    assay_features,
                    state_features,
                    perturbation_features.abs(),
                )
                perturbation_h=self.perturbation_conditioner(
                    state_h, perturbation_condition
                )
                intervention_axis_potentials=self.intervention_head(
                    (perturbation_h - state_h).mean(dim=-1)
                ).reshape(
                    h.shape[0],
                    self.config.signed_perturbation_features,
                    self.config.circuit_factors,
                )
                signed_perturbations=perturbation_features[
                    :, : self.config.signed_perturbation_features
                ]
                signed_scale=signed_perturbations.abs().sum(
                    dim=1, keepdim=True
                ).clamp_min(1.0).sqrt()
                intervention_factors=torch.einsum(
                    "bp,bpf->bf",
                    signed_perturbations,
                    intervention_axis_potentials,
                ) / signed_scale
                intervention_basis=self.intervention_basis_head(state_h)
                perturbation_residual=torch.einsum(
                    "bf,bfn->bn", intervention_factors, intervention_basis
                ) / (self.config.circuit_factors ** 0.5)
                if self.config.signed_perturbation_features:
                    perturbation_mask=(
                        signed_perturbations.abs().amax(dim=1, keepdim=True) > 0
                    ).to(perturbation_residual.dtype)
                else:
                    perturbation_mask=perturbation_residual.new_zeros(
                        (perturbation_residual.shape[0], 1)
                    )
                perturbation_residual=perturbation_residual * perturbation_mask
                if ablate_intervention_residual:
                    perturbation_residual=perturbation_residual * 0.0
                residual=state_residual + perturbation_residual

                baseline=torch.nn.functional.softplus(baseline_raw)
                mean=torch.nn.functional.softplus(baseline_raw + residual)
                combined_h=state_h + perturbation_mask[:, :, None] * (
                    perturbation_h - state_h
                )
                log_variance=self.uncertainty_head(combined_h).squeeze(1).clamp(-8.0, 8.0)
                intervention_factors=intervention_factors * perturbation_mask
                counterfactual_factors=self._counterfactual_progression_factors(
                    h, assay_features, state_features
                )
                domain_counterfactual_factors=self._domain_counterfactual_factors(
                    h, assay_features, state_features
                )
                return ChromatinOutput(
                    mean=mean,
                    baseline=baseline,
                    state_residual=state_residual,
                    perturbation_residual=perturbation_residual,
                    residual=residual,
                    log_variance=log_variance,
                    circuit_factors=circuit_factors,
                    intervention_factors=intervention_factors,
                    intervention_axis_potentials=intervention_axis_potentials,
                    counterfactual_factors=counterfactual_factors,
                    domain_counterfactual_factors=domain_counterfactual_factors,
                )

            def predict_reverse_complement_ensemble(
                self,
                sequence,
                assay_features,
                state_features,
                perturbation_features=None,
                disease_mask=None,
                ablate_state_residual=False,
                ablate_intervention_residual=False,
            ):
                direct=self(
                    sequence,
                    assay_features,
                    state_features,
                    perturbation_features,
                    disease_mask,
                    ablate_state_residual,
                    ablate_intervention_residual,
                )
                reverse=self(
                    reverse_complement_one_hot(sequence),
                    assay_features,
                    state_features,
                    perturbation_features,
                    disease_mask,
                    ablate_state_residual,
                    ablate_intervention_residual,
                )
                reverse_mean=reverse.mean.flip(-1)
                reverse_variance=reverse.log_variance.flip(-1).exp()
                ensemble_variance=(
                    (direct.log_variance.exp() + reverse_variance) / 2
                    + (direct.mean - reverse_mean).square() / 4
                ).clamp_min(1e-8)
                return ChromatinOutput(
                    mean=(direct.mean + reverse_mean) / 2,
                    baseline=(direct.baseline + reverse.baseline.flip(-1)) / 2,
                    state_residual=(
                        direct.state_residual + reverse.state_residual.flip(-1)
                    )
                    / 2,
                    perturbation_residual=(
                        direct.perturbation_residual
                        + reverse.perturbation_residual.flip(-1)
                    )
                    / 2,
                    residual=(direct.residual + reverse.residual.flip(-1)) / 2,
                    log_variance=ensemble_variance.log().clamp(-8.0, 8.0),
                    circuit_factors=(direct.circuit_factors + reverse.circuit_factors) / 2,
                    intervention_factors=(
                        direct.intervention_factors + reverse.intervention_factors
                    )
                    / 2,
                    intervention_axis_potentials=(
                        direct.intervention_axis_potentials
                        + reverse.intervention_axis_potentials
                    )
                    / 2,
                    counterfactual_factors=(
                        (direct.counterfactual_factors + reverse.counterfactual_factors) / 2
                        if direct.counterfactual_factors is not None
                        and reverse.counterfactual_factors is not None
                        else None
                    ),
                    domain_counterfactual_factors=(
                        (
                            direct.domain_counterfactual_factors
                            + reverse.domain_counterfactual_factors
                        )
                        / 2
                        if direct.domain_counterfactual_factors is not None
                        and reverse.domain_counterfactual_factors is not None
                        else None
                    ),
                )

        return _PDACircuitFormer()

class DirectConditionalCNN:

    def __new__(cls, config: ChromatinModelConfig):
        config.validate()
        if config.architecture != "direct_conditional_cnn":
            raise ValueError("DirectConditionalCNN requires its registered architecture")
        torch, nn=_torch_modules()

        class DownsampleBlock(nn.Module):
            def __init__(self, in_channels: int, out_channels: int):
                super().__init__()
                self.block=nn.Sequential(
                    nn.Conv1d(
                        in_channels, out_channels, 9, stride=2, padding=4, bias=False
                    ),
                    nn.GroupNorm(_group_count(out_channels), out_channels),
                    nn.GELU(),
                    nn.Conv1d(
                        out_channels,
                        out_channels,
                        5,
                        padding=2,
                        groups=out_channels,
                        bias=False,
                    ),
                    nn.GroupNorm(_group_count(out_channels), out_channels),
                    nn.GELU(),
                )

            def forward(self, x):
                return self.block(x)

        class GatedDilatedBlock(nn.Module):
            def __init__(self, channels: int, dilation: int):
                super().__init__()
                padding=dilation * (config.kernel_size // 2)
                self.norm=nn.GroupNorm(_group_count(channels), channels)
                self.depthwise=nn.Conv1d(
                    channels,
                    channels,
                    config.kernel_size,
                    padding=padding,
                    dilation=dilation,
                    groups=channels,
                )
                self.pointwise=nn.Conv1d(channels, channels * 2, 1)
                self.project=nn.Conv1d(channels, channels, 1)
                self.dropout=nn.Dropout(config.dropout)

            def forward(self, x):
                h=self.depthwise(self.norm(x))
                value, gate=self.pointwise(h).chunk(2, dim=1)
                return x + self.dropout(self.project(value * torch.sigmoid(gate)))

        class _DirectConditionalCNN(nn.Module):
            model_name="DirectConditionalCNN"

            def __init__(self):
                super().__init__()
                self.config=config
                channels=[
                    min(config.d_model, config.base_channels * (2 ** (stage // 2)))
                    for stage in range(config.downsample_stages)
                ]
                stem=[]
                in_channels=4
                for out_channels in channels:
                    stem.append(DownsampleBlock(in_channels, out_channels))
                    in_channels=out_channels
                self.stem=nn.ModuleList(stem)
                self.to_model=nn.Conv1d(in_channels, config.d_model, 1)
                self.blocks=nn.ModuleList(
                    GatedDilatedBlock(
                        config.d_model,
                        config.dilation_cycle[index % len(config.dilation_cycle)],
                    )
                    for index in range(config.n_layers)
                )
                condition_features=(
                    config.assay_features
                    + config.state_features
                    + config.perturbation_features
                )
                self.conditioner=nn.Sequential(
                    nn.Linear(condition_features, config.d_model),
                    nn.SiLU(),
                    nn.Linear(config.d_model, config.d_model * 2),
                )
                self.profile_head=nn.Sequential(
                    nn.GroupNorm(_group_count(config.d_model), config.d_model),
                    nn.GELU(),
                    nn.Conv1d(config.d_model, 1, 1),
                )
                self.uncertainty_head=nn.Conv1d(config.d_model, 1, 1)

            def _run_block(self, block, x):
                if self.config.gradient_checkpointing and self.training and x.requires_grad:
                    from torch.utils.checkpoint import checkpoint

                    return checkpoint(block, x, use_reentrant=False)
                return block(x)

            def encode(self, sequence):
                if sequence.ndim != 3 or sequence.shape[1] != 4:
                    raise ValueError("sequence must have shape (batch, 4, length)")
                if sequence.shape[-1] != self.config.sequence_length:
                    raise ValueError(
                        f"expected sequence length {self.config.sequence_length}, "
                        f"got {sequence.shape[-1]}"
                    )
                h=sequence
                for block in self.stem:
                    h=self._run_block(block, h)
                h=self.to_model(h)
                for block in self.blocks:
                    h=self._run_block(block, h)
                return h

            def forward(
                self,
                sequence,
                assay_features,
                state_features,
                perturbation_features=None,
                disease_mask=None,
                ablate_state_residual=False,
                ablate_intervention_residual=False,
            ):
                del disease_mask, ablate_state_residual, ablate_intervention_residual
                h=self.encode(sequence)
                if perturbation_features is None:
                    perturbation_features=h.new_zeros(
                        (h.shape[0], self.config.perturbation_features)
                    )
                conditions=torch.cat(
                    (assay_features, state_features, perturbation_features), dim=1
                )
                gamma, beta=self.conditioner(conditions).chunk(2, dim=1)
                conditioned=h * (1.0 + 0.1 * torch.tanh(gamma)[:, :, None])
                conditioned=conditioned + beta[:, :, None]
                mean=torch.nn.functional.softplus(
                    self.profile_head(conditioned).squeeze(1)
                )
                zero_profile=mean * 0.0
                zero_factors=h.new_zeros(
                    (h.shape[0], self.config.circuit_factors)
                )
                zero_axes=h.new_zeros(
                    (
                        h.shape[0],
                        self.config.signed_perturbation_features,
                        self.config.circuit_factors,
                    )
                )
                return ChromatinOutput(
                    mean=mean,
                    baseline=mean,
                    state_residual=zero_profile,
                    perturbation_residual=zero_profile,
                    residual=zero_profile,
                    log_variance=self.uncertainty_head(conditioned)
                    .squeeze(1)
                    .clamp(-8.0, 8.0),
                    circuit_factors=zero_factors,
                    intervention_factors=zero_factors,
                    intervention_axis_potentials=zero_axes,
                )

            def predict_reverse_complement_ensemble(self, sequence, *args, **kwargs):
                direct=self(sequence, *args, **kwargs)
                reverse=self(reverse_complement_one_hot(sequence), *args, **kwargs)
                reverse_mean=reverse.mean.flip(-1)
                reverse_variance=reverse.log_variance.flip(-1).exp()
                variance=(
                    (direct.log_variance.exp() + reverse_variance) / 2
                    + (direct.mean - reverse_mean).square() / 4
                ).clamp_min(1e-8)
                mean=(direct.mean + reverse_mean) / 2
                zero_profile=mean * 0.0
                return ChromatinOutput(
                    mean=mean,
                    baseline=mean,
                    state_residual=zero_profile,
                    perturbation_residual=zero_profile,
                    residual=zero_profile,
                    log_variance=variance.log().clamp(-8.0, 8.0),
                    circuit_factors=direct.circuit_factors,
                    intervention_factors=direct.intervention_factors,
                    intervention_axis_potentials=direct.intervention_axis_potentials,
                )

        return _DirectConditionalCNN()

def build_chromatin_model(config: ChromatinModelConfig):
    if config.architecture == "pdac_circuit":
        model=PDACircuitFormer(config)
        model.model_name="PDACircuitFormer"
        return model
    return DirectConditionalCNN(config)
