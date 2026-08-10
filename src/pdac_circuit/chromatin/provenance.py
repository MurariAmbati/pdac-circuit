from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path

from .streaming import sha256_file

REQUIRED_FIELDS={
    "schema",
    "model",
    "model_version",
    "prediction_bundle_sha256",
    "weights_sha256",
    "track_mapping_sha256",
    "data_snapshot_sha256",
    "command",
    "created_at",
    "training_use",
}

def validate_prediction_manifest(
    manifest_path: str | Path,
    prediction_bundle_path: str | Path,
    *,
    expected_model: str | None = None,
) -> dict:
    path=Path(manifest_path)
    failures=[]
    if not path.exists():
        return {"ok": False, "failures": [f"missing manifest {path}"]}
    payload=json.loads(path.read_text(encoding="utf-8"))
    missing=sorted(REQUIRED_FIELDS - set(payload))
    if missing:
        failures.append(f"missing required fields: {missing}")
    if payload.get("schema") != "pdac-circuit.prediction-provenance/1":
        failures.append("invalid prediction provenance schema")
    if expected_model is not None and payload.get("model") != expected_model:
        failures.append(
            f"manifest model {payload.get('model')!r} != bundle model {expected_model!r}"
        )
    try:
        datetime.fromisoformat(str(payload.get("created_at", "")).replace("Z", "+00:00"))
    except ValueError:
        failures.append("created_at is not an ISO-8601 timestamp")
    bundle_path=Path(prediction_bundle_path)
    if not bundle_path.exists():
        failures.append(f"missing prediction bundle {bundle_path}")
    elif payload.get("prediction_bundle_sha256") != sha256_file(bundle_path):
        failures.append("prediction bundle sha256 mismatch")
    for field in ("weights_sha256", "track_mapping_sha256", "data_snapshot_sha256"):
        value=str(payload.get(field, ""))
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value.lower()):
            failures.append(f"{field} must be a real 64-character sha256")
    if "raw_prediction_sha256" in payload:
        value=str(payload["raw_prediction_sha256"])
        if len(value) != 64 or any(
            char not in "0123456789abcdef" for char in value.lower()
        ):
            failures.append("raw_prediction_sha256 must be a real 64-character sha256")
    if "claim_surface_contract_sha256" in payload:
        value=str(payload["claim_surface_contract_sha256"])
        if len(value) != 64 or any(
            char not in "0123456789abcdef" for char in value.lower()
        ):
            failures.append("claim_surface_contract_sha256 must be a real sha256")
    if not str(payload.get("command", "")).strip():
        failures.append("empty prediction command")
    if payload.get("training_use") not in {"predictions_only", "candidate_model"}:
        failures.append("training_use must be predictions_only or candidate_model")
    if "seed" in payload and (
        not isinstance(payload["seed"], int) or payload["seed"] < 0
    ):
        failures.append("seed must be a non-negative integer")
    ensemble=payload.get("seed_ensemble")
    if ensemble is not None:
        if not isinstance(ensemble, dict) or ensemble.get("schema") != (
            "pdac-circuit.seed-ensemble/1"
        ):
            failures.append("invalid seed ensemble schema")
        else:
            seeds=ensemble.get("registered_seeds", [])
            components=ensemble.get("components", [])
            component_seeds=[row.get("seed") for row in components]
            if (
                len(seeds) < 3
                or len(set(seeds)) != len(seeds)
                or component_seeds != seeds
                or ensemble.get("aggregation") != "arithmetic_mean"
                or ensemble.get("exact_example_ids") is not True
            ):
                failures.append("seed ensemble policy or component order is invalid")
            for row in components:
                for field in ("raw_sha256", "weights_sha256"):
                    value=str(row.get(field, ""))
                    if len(value) != 64 or any(
                        char not in "0123456789abcdef" for char in value.lower()
                    ):
                        failures.append(f"seed ensemble component {field} is invalid")
    return {"ok": not failures, "failures": failures, "manifest": payload}

def write_prediction_manifest(
    path: str | Path,
    *,
    model: str,
    model_version: str,
    prediction_bundle_path: str | Path,
    weights_sha256: str,
    track_mapping_sha256: str,
    data_snapshot_sha256: str,
    command: str,
    training_use: str,
    source_url: str | None = None,
    seed: int | None = None,
    seed_ensemble: dict | None = None,
    raw_prediction_sha256: str | None = None,
    claim_surface_contract_sha256: str | None = None,
) -> None:
    payload={
        "schema": "pdac-circuit.prediction-provenance/1",
        "model": model,
        "model_version": model_version,
        "prediction_bundle_sha256": sha256_file(prediction_bundle_path),
        "weights_sha256": weights_sha256,
        "track_mapping_sha256": track_mapping_sha256,
        "data_snapshot_sha256": data_snapshot_sha256,
        "command": command,
        "created_at": datetime.now().astimezone().isoformat(),
        "training_use": training_use,
        "source_url": source_url,
    }
    if seed is not None:
        payload["seed"]=int(seed)
    if seed_ensemble is not None:
        payload["seed_ensemble"]=seed_ensemble
    if raw_prediction_sha256 is not None:
        payload["raw_prediction_sha256"]=raw_prediction_sha256
    if claim_surface_contract_sha256 is not None:
        payload["claim_surface_contract_sha256"]=claim_surface_contract_sha256
    report=validate_prediction_manifest.__name__
    payload["validator"]=report
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
