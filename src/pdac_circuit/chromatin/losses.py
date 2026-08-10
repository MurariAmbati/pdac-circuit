from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class LossWeights:
    profile: float = 1.0
    correlation: float = 0.25
    uncertainty: float = 0.10
    residual_delta: float = 0.50
    perturbation_delta: float = 0.50
    healthy_zero: float = 0.10
    state_graph: float = 0.05
    domain_invariance: float = 0.05

def _masked_mean(values, mask=None):
    if mask is None:
        return values.mean()
    mask=mask.to(values.dtype)
    while mask.ndim < values.ndim:
        mask=mask.unsqueeze(-1)
    denom=mask.expand_as(values).sum().clamp_min(1.0)
    return (values * mask).sum() / denom

def log_profile_loss(prediction, target, mask=None):

    import torch.nn.functional as F

    pred=prediction.clamp_min(0)
    true=target.clamp_min(0)
    losses=[]
    for factor in (1, 2, 4, 8):
        if factor > 1:
            pred=F.avg_pool1d(pred.unsqueeze(1), 2, 2).squeeze(1)
            true=F.avg_pool1d(true.unsqueeze(1), 2, 2).squeeze(1)
            local_mask=None
            if mask is not None:
                local_mask=F.avg_pool1d(mask.float().unsqueeze(1), factor, factor).squeeze(1) > 0.999
        else:
            local_mask=mask
        losses.append(_masked_mean(F.smooth_l1_loss(pred.log1p(), true.log1p(), reduction="none"), local_mask))
    return sum(losses) / len(losses)

def correlation_loss(prediction, target, mask=None, eps: float = 1e-6):

    if mask is None:
        weights=prediction.new_ones(prediction.shape)
    else:
        weights=mask.to(prediction.dtype)
    count=weights.sum(dim=-1, keepdim=True)
    safe_count=count.clamp_min(1.0)
    prediction_mean=(prediction * weights).sum(dim=-1, keepdim=True) / safe_count
    target_mean=(target * weights).sum(dim=-1, keepdim=True) / safe_count
    x=(prediction - prediction_mean) * weights
    y=(target - target_mean) * weights
    denominator=x.square().sum(dim=-1).sqrt() * y.square().sum(dim=-1).sqrt()
    valid=(count.squeeze(-1) >= 2) & (denominator > eps)
    if not valid.any():
        return prediction.sum() * 0.0
    corr=(x * y).sum(dim=-1) / denominator.clamp_min(eps)
    return 1.0 - corr[valid].mean()

def heteroscedastic_loss(prediction, target, log_variance, mask=None):

    error2=(prediction.clamp_min(0).log1p() - target.clamp_min(0).log1p()).square()
    nll=0.5 * (log_variance + error2 * (-log_variance).exp())
    return _masked_mean(nll, mask)

def paired_residual_loss(residual, paired_delta, pair_mask=None):

    import torch.nn.functional as F

    element=F.smooth_l1_loss(residual, paired_delta, reduction="none")
    return _masked_mean(element, pair_mask)

def healthy_residual_zero_loss(residual, healthy_mask):

    return _masked_mean(residual.square(), healthy_mask)

def state_graph_laplacian_loss(circuit_factors, graph_edges):

    if not graph_edges:
        return circuit_factors.sum() * 0.0
    terms=[]
    for left, right, weight in graph_edges:
        terms.append(float(weight) * (circuit_factors[left] - circuit_factors[right]).square().mean())
    return sum(terms) / len(terms)

def counterfactual_progression_loss(counterfactual_factors):

    if counterfactual_factors is None:
        return None
    if counterfactual_factors.ndim != 3 or counterfactual_factors.shape[1] != 4:
        raise ValueError("counterfactual factors must have shape (batch, 4, factors)")
    healthy, panin, primary, metastatic = counterfactual_factors.unbind(dim=1)
    first_1=panin - healthy
    first_2=primary - panin
    first_3=metastatic - primary
    curvature=(first_2 - first_1).square().mean() + (
        first_3 - first_2
    ).square().mean()
    return curvature / 2 + 0.1 * healthy.square().mean()

def domain_invariance_loss(domain_counterfactual_factors, reference):

    if domain_counterfactual_factors is None:
        return reference.sum() * 0.0
    if domain_counterfactual_factors.ndim != 3 or domain_counterfactual_factors.shape[1] != 2:
        raise ValueError("domain counterfactual factors must have shape (batch, 2, factors)")
    return (
        domain_counterfactual_factors[:, 0] - domain_counterfactual_factors[:, 1]
    ).square().mean()

def total_chromatin_loss(
    output,
    target,
    *,
    weights: LossWeights | None = None,
    signal_mask=None,
    paired_delta=None,
    perturbation_delta=None,
    pair_mask=None,
    perturbation_mask=None,
    healthy_mask=None,
    graph_edges=None,
):
    weights=weights or LossWeights()
    parts={
        "profile": log_profile_loss(output.mean, target, signal_mask),
        "correlation": correlation_loss(output.mean, target, signal_mask),
        "uncertainty": heteroscedastic_loss(
            output.mean, target, output.log_variance, signal_mask
        ),
    }
    parts["residual_delta"]=(
        paired_residual_loss(output.state_residual, paired_delta, pair_mask)
        if paired_delta is not None
        else output.mean.sum() * 0.0
    )
    parts["perturbation_delta"]=(
        paired_residual_loss(
            output.perturbation_residual, perturbation_delta, perturbation_mask
        )
        if perturbation_delta is not None
        else output.mean.sum() * 0.0
    )
    parts["healthy_zero"]=(
        healthy_residual_zero_loss(output.state_residual, healthy_mask)
        if healthy_mask is not None
        else output.mean.sum() * 0.0
    )
    counterfactual_graph=counterfactual_progression_loss(
        getattr(output, "counterfactual_factors", None)
    )
    parts["state_graph"]=(
        counterfactual_graph
        if counterfactual_graph is not None
        else state_graph_laplacian_loss(output.circuit_factors, graph_edges or [])
    )
    parts["domain_invariance"]=domain_invariance_loss(
        getattr(output, "domain_counterfactual_factors", None), output.circuit_factors
    )
    total=sum(getattr(weights, key) * value for key, value in parts.items())
    return total, parts
