from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import uuid

import numpy as np

def _sha256_file(path: Path) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()

@dataclass(frozen=True)
class CircuitInterpretationGate:

    minimum_seeds: int
    minimum_pairwise_linear_cka: float
    minimum_median_pairwise_linear_cka: float
    minimum_effective_rank: float
    maximum_single_factor_variance_fraction: float

    @classmethod
    def from_registry(cls, registry: dict) -> "CircuitInterpretationGate":
        raw=registry.get("circuit_interpretation_gate", {})
        expected={
            "minimum_seeds",
            "minimum_pairwise_linear_cka",
            "minimum_median_pairwise_linear_cka",
            "minimum_effective_rank",
            "maximum_single_factor_variance_fraction",
        }
        missing=sorted(expected - set(raw))
        if missing:
            raise ValueError(f"circuit interpretation gate is missing {missing}")
        gate=cls(**{key: raw[key] for key in expected})
        gate.validate()
        return gate

    def validate(self) -> None:
        if self.minimum_seeds < 3:
            raise ValueError("circuit interpretation requires at least three seeds")
        probabilities=(
            self.minimum_pairwise_linear_cka,
            self.minimum_median_pairwise_linear_cka,
            self.maximum_single_factor_variance_fraction,
        )
        if any(not 0.0 <= value <= 1.0 for value in probabilities):
            raise ValueError("CKA and variance-fraction thresholds must be in [0, 1]")
        if (
            self.minimum_median_pairwise_linear_cka
            < self.minimum_pairwise_linear_cka
        ):
            raise ValueError("median CKA threshold cannot be below minimum pairwise CKA")
        if self.minimum_effective_rank <= 1.0:
            raise ValueError("minimum effective rank must exceed one")

def _raw_factor_bundle(path: str | Path) -> dict:
    source=Path(path)
    with np.load(source, allow_pickle=False) as bundle:
        required={"model", "example_id", "prediction", "metadata"}
        missing=sorted(required - set(bundle.files))
        if missing:
            raise ValueError(f"{source.name} is missing raw prediction fields {missing}")
        example_id=bundle["example_id"].astype(str)
        factors=bundle["prediction"].astype(np.float64)
        model=str(bundle["model"].item())
        metadata=json.loads(str(bundle["metadata"].item()))
    if metadata.get("schema") != "pdac-circuit.raw-predictions/1":
        raise ValueError(f"{source.name} is not a raw candidate prediction bundle")
    if metadata.get("component") not in {"circuit_factors", "intervention_factors"}:
        raise ValueError(
            f"{source.name} component must be circuit_factors or intervention_factors"
        )
    if factors.ndim != 2 or factors.shape[0] != len(example_id) or factors.shape[1] < 2:
        raise ValueError(f"{source.name} factors must have shape (examples, factors>=2)")
    if len(example_id) == 0 or len(set(example_id)) != len(example_id):
        raise ValueError(f"{source.name} example IDs are empty or duplicated")
    if not np.isfinite(factors).all():
        raise ValueError(f"{source.name} contains non-finite factors")
    seed=metadata.get("seed")
    if not isinstance(seed, int) or seed < 0:
        raise ValueError(f"{source.name} does not record one non-negative integer seed")
    return {
        "path": str(source),
        "sha256": _sha256_file(source),
        "model": model,
        "component": metadata["component"],
        "seed": seed,
        "example_id": example_id,
        "factors": factors,
    }

def _center(factors: np.ndarray) -> np.ndarray:
    return factors - factors.mean(axis=0, keepdims=True)

def _representation_summary(factors: np.ndarray) -> dict[str, float]:
    centered=_center(factors)
    singular=np.linalg.svd(centered, compute_uv=False)
    variance=np.square(singular)
    total=float(variance.sum())
    if total <= np.finfo(np.float64).eps:
        return {
            "effective_rank": 0.0,
            "single_factor_variance_fraction": 1.0,
        }
    probability=variance / total
    positive=probability > 0
    entropy=-float(np.sum(probability[positive] * np.log(probability[positive])))
    return {
        "effective_rank": float(np.exp(entropy)),
        "single_factor_variance_fraction": float(probability.max()),
    }

def _linear_cka(left: np.ndarray, right: np.ndarray) -> float:
    left_centered=_center(left)
    right_centered=_center(right)
    cross=left_centered.T @ right_centered
    left_self=left_centered.T @ left_centered
    right_self=right_centered.T @ right_centered
    numerator=float(np.square(cross).sum())
    denominator=float(
        np.sqrt(np.square(left_self).sum() * np.square(right_self).sum())
    )
    if denominator <= np.finfo(np.float64).eps:
        return 0.0
    return float(np.clip(numerator / denominator, 0.0, 1.0))

def _orthogonal_procrustes_nrmse(left: np.ndarray, right: np.ndarray) -> float:
    left_centered=_center(left)
    right_centered=_center(right)
    u, _, vt=np.linalg.svd(left_centered.T @ right_centered, full_matrices=False)
    aligned=left_centered @ (u @ vt)
    denominator=float(np.linalg.norm(right_centered))
    if denominator <= np.finfo(np.float64).eps:
        return float("inf")
    return float(np.linalg.norm(aligned - right_centered) / denominator)

def audit_circuit_stability(
    paths: list[str | Path], registry: dict
) -> dict:

    gate=CircuitInterpretationGate.from_registry(registry)
    bundles=[_raw_factor_bundle(path) for path in sorted(map(Path, paths))]
    if len(bundles) < gate.minimum_seeds:
        raise ValueError(
            f"circuit audit requires {gate.minimum_seeds} seeds, got {len(bundles)}"
        )
    seeds=[bundle["seed"] for bundle in bundles]
    if len(set(seeds)) != len(seeds):
        raise ValueError("circuit audit seed identities must be unique")
    models={bundle["model"] for bundle in bundles}
    components={bundle["component"] for bundle in bundles}
    dimensions={bundle["factors"].shape[1] for bundle in bundles}
    if len(models) != 1 or len(components) != 1 or len(dimensions) != 1:
        raise ValueError("circuit audit bundles differ in model, component, or factor width")

    reference_ids=bundles[0]["example_id"]
    reference_set=set(reference_ids)
    ordered_factors=[]
    per_seed=[]
    for bundle in bundles:
        observed=set(bundle["example_id"])
        if observed != reference_set:
            missing=sorted(reference_set - observed)
            extra=sorted(observed - reference_set)
            raise ValueError(
                f"seed {bundle['seed']} cohort drift: missing={missing[:5]}, extra={extra[:5]}"
            )
        index={example_id: i for i, example_id in enumerate(bundle["example_id"])}
        factors=bundle["factors"][[index[example_id] for example_id in reference_ids]]
        ordered_factors.append(factors)
        per_seed.append({"seed": bundle["seed"], **_representation_summary(factors)})

    pairwise=[]
    for left_index in range(len(bundles)):
        for right_index in range(left_index + 1, len(bundles)):
            pairwise.append(
                {
                    "left_seed": bundles[left_index]["seed"],
                    "right_seed": bundles[right_index]["seed"],
                    "linear_cka": _linear_cka(
                        ordered_factors[left_index], ordered_factors[right_index]
                    ),
                    "orthogonal_procrustes_nrmse": _orthogonal_procrustes_nrmse(
                        ordered_factors[left_index], ordered_factors[right_index]
                    ),
                }
            )
    cka=np.asarray([row["linear_cka"] for row in pairwise], dtype=np.float64)
    minimum_rank=min(row["effective_rank"] for row in per_seed)
    maximum_fraction=max(
        row["single_factor_variance_fraction"] for row in per_seed
    )
    checks={
        "minimum_seed_count": len(bundles) >= gate.minimum_seeds,
        "minimum_pairwise_linear_cka": float(cka.min())
        >= gate.minimum_pairwise_linear_cka,
        "median_pairwise_linear_cka": float(np.median(cka))
        >= gate.minimum_median_pairwise_linear_cka,
        "minimum_effective_rank": minimum_rank >= gate.minimum_effective_rank,
        "maximum_single_factor_variance_fraction": maximum_fraction
        <= gate.maximum_single_factor_variance_fraction,
    }
    return {
        "schema": "pdac-circuit.circuit-stability-audit/1",
        "model": next(iter(models)),
        "component": next(iter(components)),
        "examples": len(reference_ids),
        "factors": next(iter(dimensions)),
        "seeds": seeds,
        "inputs": [
            {"path": bundle["path"], "sha256": bundle["sha256"]}
            for bundle in bundles
        ],
        "gate": gate.__dict__,
        "per_seed": per_seed,
        "pairwise": pairwise,
        "summary": {
            "minimum_pairwise_linear_cka": float(cka.min()),
            "median_pairwise_linear_cka": float(np.median(cka)),
            "minimum_effective_rank": minimum_rank,
            "maximum_single_factor_variance_fraction": maximum_fraction,
        },
        "checks": checks,
        "interpretation_status": "PASS" if all(checks.values()) else "ABSTAIN",
        "coordinate_identifiability_claimed": False,
    }

def write_circuit_stability_audit(path: str | Path, report: dict) -> None:
    destination=Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary=destination.parent / f".{destination.name}.partial-{uuid.uuid4().hex}"
    temporary.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(destination)
