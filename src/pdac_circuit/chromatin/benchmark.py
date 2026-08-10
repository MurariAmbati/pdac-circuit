from __future__ import annotations

from dataclasses import asdict,dataclass
import json
from pathlib import Path
from typing import Callable

import numpy as np

@dataclass(frozen=True)
class PredictionBundle:
    model: str
    example_id: np.ndarray
    target: np.ndarray
    prediction: np.ndarray
    group: np.ndarray
    split: np.ndarray
    lower: np.ndarray | None = None
    upper: np.ndarray | None = None
    mask: np.ndarray | None = None

    def validate(self) -> None:
        n = len(self.example_id)
        if n == 0:
            raise ValueError("prediction bundle is empty")
        for name in ("target","prediction","group","split"):
            value = getattr(self,name)
            if len(value) != n:
                raise ValueError(f"{name} length {len(value)} != example_id length {n}")
        if self.target.shape != self.prediction.shape:
            raise ValueError("target and prediction shapes differ")
        if len(set(map(str,self.example_id))) != n:
            raise ValueError("example_id values must be unique")
        if not np.isfinite(self.target).all() or not np.isfinite(self.prediction).all():
            raise ValueError("target and prediction must be finite")
        if (self.lower is None) != (self.upper is None):
            raise ValueError("lower and upper intervals must be supplied together")
        if self.lower is not None and (
            self.lower.shape != self.target.shape or self.upper.shape != self.target.shape
        ):
            raise ValueError("interval arrays must match target shape")
        if self.mask is not None and self.mask.shape != self.target.shape:
            raise ValueError("validity mask must match target shape")

@dataclass(frozen=True)
class BenchmarkRule:
    axis: str
    metric: str
    minimum_delta: float
    minimum_groups: int = 5
    confidence: float = 0.95
    higher_is_better: bool = True
    require_ci_above_zero: bool = True
    required_for_claim: bool = True

@dataclass(frozen=True)
class AxisResult:
    axis: str
    metric: str
    candidate: float
    baseline: float
    delta: float
    ci_low: float
    ci_high: float
    one_sided_p: float
    independent_groups: int
    passed: bool
    reason: str

def save_prediction_bundle(bundle: PredictionBundle,path: str | Path) -> None:
    bundle.validate()
    payload = {
        "model": np.asarray(bundle.model),
        "example_id": bundle.example_id,
        "target": bundle.target,
        "prediction": bundle.prediction,
        "group": bundle.group,
        "split": bundle.split,
    }
    if bundle.lower is not None:
        payload["lower"] = bundle.lower
        payload["upper"] = bundle.upper
    if bundle.mask is not None:
        payload["mask"] = bundle.mask.astype(np.uint8)
    np.savez_compressed(path,**payload)

def load_prediction_bundle(path: str | Path) -> PredictionBundle:
    with np.load(path,allow_pickle=False) as payload:
        keys = set(payload.files)
        required = {"model","example_id","target","prediction","group","split"}
        missing = sorted(required - keys)
        if missing:
            raise ValueError(f"prediction bundle missing keys: {missing}")
        bundle = PredictionBundle(
            model=str(payload["model"].item()),
            example_id=payload["example_id"].copy(),
            target=payload["target"].astype(np.float64),
            prediction=payload["prediction"].astype(np.float64),
            group=payload["group"].copy(),
            split=payload["split"].copy(),
            lower=payload["lower"].astype(np.float64) if "lower" in keys else None,
            upper=payload["upper"].astype(np.float64) if "upper" in keys else None,
            mask=payload["mask"].astype(bool) if "mask" in keys else None,
        )
    bundle.validate()
    return bundle

def _pearson(y: np.ndarray,p: np.ndarray) -> float:
    y,p = y.reshape(-1),p.reshape(-1)
    if y.size < 2 or np.std(y) == 0 or np.std(p) == 0:
        return float("nan")
    return float(np.corrcoef(y,p)[0,1])

def _spearman(y: np.ndarray,p: np.ndarray) -> float:
    from scipy.stats import rankdata

    return _pearson(rankdata(y.reshape(-1)),rankdata(p.reshape(-1)))

def _average_precision(y: np.ndarray,p: np.ndarray) -> float:
    from sklearn.metrics import average_precision_score

    y = y.reshape(-1)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(average_precision_score(y,p.reshape(-1)))

def _direction_accuracy(y: np.ndarray,p: np.ndarray) -> float:
    y,p = y.reshape(-1),p.reshape(-1)
    informative = y != 0
    return float(np.mean(np.sign(y[informative]) == np.sign(p[informative]))) if informative.any() else float("nan")

def _negative_log_mae(y: np.ndarray,p: np.ndarray) -> float:
    return -float(np.mean(np.abs(np.log1p(np.clip(y,0,None)) - np.log1p(np.clip(p,0,None)))))

def _negative_asinh_mae(y: np.ndarray,p: np.ndarray) -> float:

    return -float(np.mean(np.abs(np.arcsinh(y) - np.arcsinh(p))))

METRICS: dict[str,Callable[[np.ndarray,np.ndarray],float]] = {
    "pearson": _pearson,
    "spearman": _spearman,
    "average_precision": _average_precision,
    "direction_accuracy": _direction_accuracy,
    "negative_log_mae": _negative_log_mae,
    "negative_asinh_mae": _negative_asinh_mae,
}

def align_bundles(candidate: PredictionBundle,baseline: PredictionBundle):
    candidate.validate()
    baseline.validate()
    baseline_index = {str(key): i for i,key in enumerate(baseline.example_id)}
    missing = [str(key) for key in candidate.example_id if str(key) not in baseline_index]
    if missing:
        raise ValueError(f"baseline lacks {len(missing)} candidate example IDs")
    if len(baseline_index) != len(candidate.example_id):
        raise ValueError(
            "candidate and baseline must contain exactly the same example IDs"
        )
    order = np.asarray([baseline_index[str(key)] for key in candidate.example_id])
    if not np.allclose(candidate.target,baseline.target[order],rtol=1e-5,atol=1e-7):
        raise ValueError("candidate and baseline targets differ after example alignment")
    if not np.array_equal(
        candidate.group.astype(str),baseline.group[order].astype(str)
    ):
        raise ValueError("candidate and baseline independent-group labels differ")
    if not np.array_equal(
        candidate.split.astype(str),baseline.split[order].astype(str)
    ):
        raise ValueError("candidate and baseline split labels differ")
    if (candidate.mask is None) != (baseline.mask is None):
        raise ValueError("candidate and baseline validity-mask presence differs")
    if candidate.mask is not None and not np.array_equal(candidate.mask,baseline.mask[order]):
        raise ValueError("candidate and baseline validity masks differ after alignment")
    return order

def grouped_scores(bundle: PredictionBundle,metric: str,*,split: str) -> dict[str,float]:
    if metric not in METRICS:
        raise ValueError(f"unsupported metric {metric!r}; choose from {sorted(METRICS)}")
    fn = METRICS[metric]
    selected = bundle.split.astype(str) == split
    scores = {}
    for group in sorted(set(bundle.group[selected].astype(str))):
        rows = selected & (bundle.group.astype(str) == group)
        target = bundle.target[rows]
        prediction = bundle.prediction[rows]
        if bundle.mask is not None:
            valid = bundle.mask[rows]
            target = target[valid]
            prediction = prediction[valid]
        score = fn(target,prediction) if target.size else float("nan")
        if np.isfinite(score):
            scores[group] = score
    return scores

def validate_registered_axis_groups(
    bundle: PredictionBundle,
    *,
    split: str,
    allowed_groups: list[str] | None = None,
    exact_groups_required: bool = False,
) -> dict:

    bundle.validate()
    selected = bundle.split.astype(str) == split
    observed = sorted(set(bundle.group[selected].astype(str)))
    if not observed:
        raise ValueError(f"prediction bundle has no groups for split {split!r}")
    if allowed_groups is None:
        if exact_groups_required:
            raise ValueError("exact group enforcement requires allowed_groups")
        return {"ok": True,"observed_groups": observed,"registered": False}
    registered = [str(value) for value in allowed_groups]
    if not registered or len(registered) != len(set(registered)):
        raise ValueError("allowed_groups must be non-empty and unique")
    unexpected = sorted(set(observed) - set(registered))
    missing = sorted(set(registered) - set(observed))
    if unexpected or (exact_groups_required and missing):
        raise ValueError(
            "registered independent-group contract failed: "
            f"missing={missing}, unexpected={unexpected}"
        )
    return {
        "ok": True,
        "observed_groups": observed,
        "allowed_groups": registered,
        "exact_groups_required": bool(exact_groups_required),
        "registered": True,
    }

def compare_axis(
    candidate: PredictionBundle,
    baseline: PredictionBundle,
    rule: BenchmarkRule,
    *,
    split: str,
    bootstrap: int = 10_000,
    seed: int = 20_260_620,
) -> AxisResult:
    order = align_bundles(candidate,baseline)
    aligned_baseline = PredictionBundle(
        baseline.model,
        baseline.example_id[order],
        baseline.target[order],
        baseline.prediction[order],
        baseline.group[order],
        baseline.split[order],
        baseline.lower[order] if baseline.lower is not None else None,
        baseline.upper[order] if baseline.upper is not None else None,
        baseline.mask[order] if baseline.mask is not None else None,
    )
    cand_scores = grouped_scores(candidate,rule.metric,split=split)
    base_scores = grouped_scores(aligned_baseline,rule.metric,split=split)
    groups = sorted(set(cand_scores) & set(base_scores))
    if len(groups) < rule.minimum_groups:
        return AxisResult(
            rule.axis,
            rule.metric,
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            float("nan"),
            1.0,
            len(groups),
            False,
            f"only {len(groups)} independent groups; require {rule.minimum_groups}",
        )

    direction = 1.0 if rule.higher_is_better else -1.0
    cand = np.asarray([cand_scores[g] for g in groups])
    base = np.asarray([base_scores[g] for g in groups])
    deltas = direction * (cand - base)
    rng = np.random.default_rng(seed)
    indices = rng.integers(0,len(groups),size=(bootstrap,len(groups)))
    draws = deltas[indices].mean(axis=1)
    alpha = 1.0 - rule.confidence
    ci_low,ci_high = np.quantile(draws,[alpha / 2,1 - alpha / 2])
    delta = float(deltas.mean())
    p = float((1 + np.sum(draws <= 0)) / (bootstrap + 1))
    ci_ok = bool(ci_low > 0) if rule.require_ci_above_zero else True
    passed = bool(delta >= rule.minimum_delta and ci_ok)
    reason = "pass" if passed else (
        f"delta {delta:.4g} below {rule.minimum_delta:.4g}"
        if delta < rule.minimum_delta
        else f"paired group-bootstrap CI crosses zero ({ci_low:.4g}, {ci_high:.4g})"
    )
    return AxisResult(
        rule.axis,
        rule.metric,
        float(cand.mean()),
        float(base.mean()),
        delta,
        float(ci_low),
        float(ci_high),
        p,
        len(groups),
        passed,
        reason,
    )

def interval_calibration(
    bundle: PredictionBundle,
    *,
    split: str,
    nominal: float = 0.90,
    minimum_groups: int = 5,
    max_width_iqr_multiplier: float = 4.0,
) -> dict:
    bundle.validate()
    if bundle.lower is None:
        return {"ok": False,"reason": "prediction intervals absent"}
    mask = bundle.split.astype(str) == split
    if not mask.any():
        return {"ok": False,"reason": f"no examples for split {split}"}
    covered = (bundle.target[mask] >= bundle.lower[mask]) & (
        bundle.target[mask] <= bundle.upper[mask]
    )
    valid = (
        bundle.mask[mask].astype(bool)
        if bundle.mask is not None
        else np.ones_like(covered,dtype=bool)
    )
    selected_groups = bundle.group[mask].astype(str)
    group_coverages = {}
    for group in sorted(set(selected_groups)):
        group_rows = selected_groups == group
        group_valid = valid[group_rows]
        if group_valid.any():
            group_coverages[group] = float(covered[group_rows][group_valid].mean())
    group_count = len(group_coverages)
    if group_count == 0:
        return {"ok": False,"reason": "no valid interval targets in selected groups"}
    coverage = float(np.mean(list(group_coverages.values())))
    width = float(np.mean((bundle.upper[mask] - bundle.lower[mask])[valid]))
    flattened_target = bundle.target[mask][valid]
    target_iqr = float(
        np.quantile(flattened_target,0.75) - np.quantile(flattened_target,0.25)
    )
    if target_iqr <= 1e-12:
        target_iqr = max(float(np.std(flattened_target)),1e-12)
    width_iqr_ratio = width / target_iqr
    tolerance = max(
        0.02,
        1.96 * np.sqrt(nominal * (1 - nominal) / max(group_count,1)),
    )
    coverage_ok = coverage >= nominal - tolerance
    groups_ok = group_count >= minimum_groups
    sharpness_ok = width_iqr_ratio <= max_width_iqr_multiplier
    return {
        "ok": bool(coverage_ok and groups_ok and sharpness_ok),
        "coverage": coverage,
        "nominal": nominal,
        "tolerance": tolerance,
        "mean_width": width,
        "target_iqr": target_iqr,
        "width_iqr_ratio": width_iqr_ratio,
        "max_width_iqr_multiplier": max_width_iqr_multiplier,
        "independent_groups": group_count,
        "minimum_groups": minimum_groups,
        "group_coverages": group_coverages,
        "coverage_ok": bool(coverage_ok),
        "groups_ok": bool(groups_ok),
        "sharpness_ok": bool(sharpness_ok),
    }

def claim_report(
    candidate_model: str,
    baseline_model: str,
    axis_results: list[AxisResult],
    calibration: dict,
    rules: list[BenchmarkRule],
    provenance: dict | None = None,
    seed_robustness: dict | None = None,
    comparison_role: str | None = None,
    model_identity_policy: dict | None = None,
) -> dict:
    required_axes = {rule.axis for rule in rules if rule.required_for_claim}
    seen = {result.axis for result in axis_results}
    missing = sorted(required_axes - seen)
    failed = [result.axis for result in axis_results if result.axis in required_axes and not result.passed]
    provenance_ok = bool(provenance and provenance.get("ok"))
    seed_ok = bool(seed_robustness is None or seed_robustness.get("ok"))
    beats = (
        not missing
        and not failed
        and bool(calibration.get("ok"))
        and provenance_ok
        and seed_ok
    )
    return {
        "schema": "pdac-circuit.enformer-benchmark/1",
        "candidate": candidate_model,
        "baseline": baseline_model,
        "comparison_role": comparison_role,
        "model_identity_policy": model_identity_policy,
        "verdict": "BEATS_BASELINE" if beats else "ABSTAIN",
        "missing_required_axes": missing,
        "failed_required_axes": failed,
        "calibration": calibration,
        "provenance": provenance or {"ok": False,"reason": "prediction provenance absent"},
        "seed_robustness": seed_robustness
        or {"ok": True,"required": False,"reason": "not registered"},
        "axes": [asdict(result) for result in axis_results],
        "rules": [asdict(rule) for rule in rules],
        "ruo": True,
    }

def write_claim_report(report: dict,path: str | Path) -> None:
    Path(path).write_text(json.dumps(report,indent=2,sort_keys=True),encoding="utf-8")

def claim_suite_report(registry: dict,reports: dict[str,dict]) -> dict:

    policy = registry.get("claim_suite_policy",{})
    if policy.get("schema") != "pdac-circuit.chromatin-claim-suite-policy/1":
        raise ValueError("missing or invalid claim-suite policy")
    required_roles = list(policy.get("required_roles",[]))
    secondary_roles = list(policy.get("reported_secondary_roles",[]))
    expected_roles = required_roles + secondary_roles
    if list(reports) != expected_roles:
        raise ValueError(
            f"claim-suite roles {list(reports)} differ from registered {expected_roles}"
        )
    expected_axes = [row["axis"] for row in registry.get("rules",[])]
    comparison_policy = registry.get("comparison_model_policy",{})
    failures: list[str] = []
    role_status = {}
    candidate_identity = None
    for role,report in reports.items():
        if report.get("schema") != "pdac-circuit.enformer-benchmark/1":
            raise ValueError(f"{role} has an invalid benchmark schema")
        if report.get("comparison_role") != role:
            raise ValueError(f"{role} report is labeled {report.get('comparison_role')!r}")
        expected_identity = comparison_policy.get(role,{})
        if (
            report.get("candidate") != expected_identity.get("candidate_model")
            or report.get("baseline") != expected_identity.get("baseline_model")
        ):
            raise ValueError(f"{role} model identity differs from the registry")
        axes = [row.get("axis") for row in report.get("axes",[])]
        rules = [row.get("axis") for row in report.get("rules",[])]
        if axes != expected_axes or rules != expected_axes:
            raise ValueError(f"{role} benchmark axes/order differ from the registry")
        provenance_axes = report.get("provenance",{}).get("axes",{})
        identity = []
        for axis in expected_axes:
            manifest = (
                provenance_axes.get(axis,{}).get("candidate",{}).get("manifest",{})
            )
            identity.append(
                (
                    axis,
                    manifest.get("prediction_bundle_sha256"),
                    manifest.get("weights_sha256"),
                    manifest.get("claim_surface_contract_sha256"),
                    json.dumps(
                        manifest.get("seed_ensemble"),
                        sort_keys=True,
                        separators=(",",":"),
                    ),
                )
            )
        if any(not all(row[1:4]) for row in identity):
            failures.append(f"{role} lacks a complete candidate bundle identity")
        if candidate_identity is None:
            candidate_identity = identity
        elif identity != candidate_identity:
            failures.append(f"{role} candidate bundles differ from the headline candidate")
        passed = report.get("verdict") == "BEATS_BASELINE"
        role_status[role] = {
            "verdict": report.get("verdict"),
            "passed": passed,
            "required_for_official_verdict": role in required_roles,
            "failed_required_axes": report.get("failed_required_axes",[]),
            "missing_required_axes": report.get("missing_required_axes",[]),
            "baseline": report.get("baseline"),
        }
        if role in required_roles and not passed:
            failures.append(f"required comparison {role} did not beat its frozen baseline")
    official_pass = not failures
    return {
        "schema": "pdac-circuit.chromatin-claim-suite/1",
        "verdict": policy["official_verdict"] if official_pass else policy["failure_status"],
        "candidate": comparison_policy[required_roles[0]]["candidate_model"],
        "required_roles": required_roles,
        "reported_secondary_roles": secondary_roles,
        "candidate_identity_equal_across_roles": not any(
            "candidate bundles differ" in failure for failure in failures
        ),
        "roles": role_status,
        "failures": failures,
        "ruo": True,
    }
