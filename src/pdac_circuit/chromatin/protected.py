from __future__ import annotations

from dataclasses import asdict, replace
from datetime import datetime
import json
import math
from pathlib import Path
import uuid

from .streaming import sha256_file

SEAL_SCHEMA="pdac-circuit.protected-study-seal/1"
RELEASE_SCHEMA="pdac-circuit.protected-study-release/1"

def _resolve(root: Path, value: str | Path) -> Path:
    path=Path(value)
    return path if path.is_absolute() else root / path

def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary=path.parent / f".{path.name}.partial-{uuid.uuid4().hex}"
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)

def freeze_protected_study_seal(
    project_root: str | Path,
    *,
    campaign_path: str | Path,
    registry_path: str | Path,
    assets_path: str | Path,
    out: str | Path,
) -> dict:

    root=Path(project_root).resolve()
    campaign_file=_resolve(root, campaign_path)
    registry_file=_resolve(root, registry_path)
    assets_file=_resolve(root, assets_path)
    campaign=json.loads(campaign_file.read_text(encoding="utf-8"))
    registry=json.loads(registry_file.read_text(encoding="utf-8"))
    assets=json.loads(assets_file.read_text(encoding="utf-8"))
    if campaign.get("selection", {}).get("maximum_tuning_uses_of_test_surfaces") != 0:
        raise ValueError("campaign permits protected test-surface tuning")
    policy=registry.get("protected_study_policy", {})
    registered_planes=policy.get("planes", {})
    planes=[
        row
        for row in campaign.get("data_planes", [])
        if row.get("protected_from_training") is True
    ]
    observed_planes={
        str(row.get("name")): [str(value).upper() for value in row.get("sources", [])]
        for row in planes
    }
    if observed_planes != registered_planes:
        raise ValueError("campaign protected data planes differ from the registry policy")
    accessions=[
        accession
        for plane_accessions in observed_planes.values()
        for accession in plane_accessions
    ]
    if len(accessions) < 2 or len(set(accessions)) != len(accessions):
        raise ValueError("protected external accessions are incomplete or duplicated")
    if accessions != policy.get("accessions"):
        raise ValueError("protected accession order differs from the registry policy")
    from .claim_surfaces import validate_claim_surface_contract

    claim_surface_report=validate_claim_surface_contract(
        root,
        registry_path=registry_file,
        assets_path=assets_file,
        campaign_path=campaign_file,
        contract_path=policy.get("claim_surface_contract", ""),
    )
    accession_planes={
        accession: plane
        for plane, plane_accessions in observed_planes.items()
        for accession in plane_accessions
    }
    asset_rows={str(row.get("id", "")).upper(): row for row in assets.get("assets", [])}
    studies=[]
    early_access=[]
    for accession in accessions:
        asset=asset_rows.get(accession)
        if not asset or "test" not in str(asset.get("split", "")).lower():
            raise ValueError(f"{accession} is not a registered external test asset")
        plan_path=root / "data" / "manifests" / "studies" / f"{accession}.plan.json"
        plan=json.loads(plan_path.read_text(encoding="utf-8"))
        if (
            plan.get("schema") != "pdac-circuit.geo-download-plan/1"
            or plan.get("accession") != accession
            or plan.get("protected_from_training") is not True
        ):
            raise ValueError(f"{accession} protected download plan is invalid")
        protected_dir=root / "data" / "studies" / "protected" / accession
        metadata_dir=root / "data" / "metadata" / "geo" / accession
        if protected_dir.exists() or metadata_dir.exists():
            early_access.append(accession)
        studies.append(
            {
                "accession": accession,
                "plane": accession_planes[accession],
                "role": asset.get("role"),
                "split": asset.get("split"),
                "plan": str(plan_path.relative_to(root)),
                "plan_sha256": sha256_file(plan_path),
                "registered_files": [
                    {
                        "name": row.get("name"),
                        "bytes": row.get("bytes"),
                        "url": row.get("url"),
                    }
                    for row in plan.get("files", [])
                ],
            }
        )
    if early_access:
        raise RuntimeError(
            f"cannot create a pre-access seal after protected paths exist: {early_access}"
        )
    if registry.get("candidate_seed_policy", {}).get("registered_seeds") != campaign.get(
        "seeds"
    ):
        raise ValueError("registry and campaign seed policies differ")
    payload={
        "schema": SEAL_SCHEMA,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "SEALED_BEFORE_ACCESS",
        "campaign": str(campaign_file.relative_to(root)),
        "campaign_sha256": sha256_file(campaign_file),
        "registry": str(registry_file.relative_to(root)),
        "registry_sha256": sha256_file(registry_file),
        "assets": str(assets_file.relative_to(root)),
        "assets_sha256": sha256_file(assets_file),
        "claim_surface_contract": claim_surface_report["contract"],
        "claim_surface_contract_sha256": claim_surface_report["sha256"],
        "protected_planes": observed_planes,
        "design_metadata_policy": policy.get("design_metadata_policy"),
        "registered_seeds": campaign["seeds"],
        "test_tuning_uses_at_seal": 0,
        "protected_download_present_at_seal": False,
        "protected_metadata_present_at_seal": False,
        "metadata_release_required": True,
        "target_download_release_required": True,
        "studies": studies,
    }
    destination=_resolve(root, out)
    _atomic_json(destination, payload)
    return {
        "out": str(destination),
        "sha256": sha256_file(destination),
        "accessions": accessions,
        "status": payload["status"],
    }

def validate_protected_study_seal(
    project_root: str | Path,
    seal_path: str | Path,
    *,
    accession: str | None = None,
) -> dict:
    root=Path(project_root).resolve()
    path=_resolve(root, seal_path)
    payload=json.loads(path.read_text(encoding="utf-8"))
    failures=[]
    if payload.get("schema") != SEAL_SCHEMA or payload.get("status") != (
        "SEALED_BEFORE_ACCESS"
    ):
        failures.append("invalid protected-study seal schema/status")
    for field in ("campaign", "registry", "assets", "claim_surface_contract"):
        source=_resolve(root, payload.get(field, ""))
        if not source.is_file() or payload.get(f"{field}_sha256") != sha256_file(source):
            failures.append(f"protected-study seal {field} hash drifted")
    studies=payload.get("studies", [])
    registered={row.get("accession") for row in studies}
    if accession is not None and accession.upper() not in registered:
        failures.append(f"{accession.upper()} is absent from the protected-study seal")
    for row in studies:
        plan=_resolve(root, row.get("plan", ""))
        if not plan.is_file() or row.get("plan_sha256") != sha256_file(plan):
            failures.append(f"protected plan hash drifted for {row.get('accession')}")
    if payload.get("metadata_release_required") is not True:
        failures.append("protected-study seal does not require a metadata release")
    if payload.get("target_download_release_required") is not True:
        failures.append("protected-study seal does not require a target-download release")
    return {
        "ok": not failures,
        "failures": failures,
        "seal": payload,
        "path": str(path),
        "sha256": sha256_file(path),
    }

def authorize_protected_metadata_release(
    project_root: str | Path,
    *,
    seal_path: str | Path,
    checkpoint_paths: list[str | Path],
    out: str | Path,
) -> dict:

    import torch

    from .config import ChromatinModelConfig, ChromatinTrainConfig, load_chromatin_config
    from .trainer import _code_fingerprint, _config_hash

    root=Path(project_root).resolve()
    seal_report=validate_protected_study_seal(root, seal_path)
    if not seal_report["ok"]:
        raise ValueError(f"protected-study seal failed: {seal_report['failures']}")
    required_seeds=[int(seed) for seed in seal_report["seal"]["registered_seeds"]]
    if len(checkpoint_paths) != len(required_seeds):
        raise ValueError(
            f"received {len(checkpoint_paths)} checkpoints; require {len(required_seeds)}"
        )
    records={}
    model_config=None
    code_fingerprint=None
    campaign_path=_resolve(root, seal_report["seal"]["campaign"])
    campaign=json.loads(campaign_path.read_text(encoding="utf-8"))
    checkpoint_root=_resolve(
        root,
        campaign.get("execution", {}).get(
            "checkpoint_root", "models/chromatin/campaign"
        ),
    ).resolve()
    main_profiles={
        Path(row["config"]).stem: row["config"]
        for row in campaign.get("profiles", [])
    }
    selected_profile=None
    for value in checkpoint_paths:
        path=_resolve(root, value)
        if path.name != "best.pt" or not path.is_file():
            raise ValueError(f"protected release requires existing best.pt checkpoints: {path}")
        try:
            relative_checkpoint=path.resolve().relative_to(checkpoint_root)
        except ValueError as exc:
            raise ValueError(f"{path} is outside the registered campaign checkpoint root") from exc
        parts=relative_checkpoint.parts
        if (
            len(parts) != 4
            or parts[0] not in main_profiles
            or not parts[1].startswith("seed-")
            or parts[2] != "04-human_state_adaptation"
            or parts[3] != "best.pt"
        ):
            raise ValueError(f"{path} is not a registered main-profile final best checkpoint")
        profile_slug=parts[0]
        if selected_profile is None:
            selected_profile=profile_slug
        elif profile_slug != selected_profile:
            raise ValueError("protected release seed checkpoints use different campaign profiles")
        state=torch.load(path, map_location="cpu", weights_only=False)
        if (
            state.get("schema") != "pdac-circuit.chromatin-checkpoint/1"
            or state.get("training_stage") != "human_state_adaptation"
        ):
            raise ValueError(f"{path} is not a final human_state_adaptation checkpoint")
        raw_model_config=dict(state.get("model_config", {}))
        if "dilation_cycle" in raw_model_config:
            raw_model_config["dilation_cycle"]=tuple(raw_model_config["dilation_cycle"])
        candidate_model_config=ChromatinModelConfig(**raw_model_config)
        candidate_train_config=ChromatinTrainConfig(**state.get("train_config", {}))
        candidate_model_config.validate()
        candidate_train_config.validate()
        expected_config_hash=_config_hash(
            candidate_model_config,
            candidate_train_config,
            "human_state_adaptation",
        )
        if state.get("config_hash") != expected_config_hash:
            raise ValueError(f"{path} configuration hash drifted")
        if state.get("code_fingerprint") != _code_fingerprint():
            raise ValueError(f"{path} behavioral code fingerprint is stale")
        seed=state.get("train_config", {}).get("seed")
        if not isinstance(seed, int) or seed not in required_seeds or seed in records:
            raise ValueError(f"{path} has an invalid or duplicate campaign seed {seed}")
        if parts[1] != f"seed-{seed}":
            raise ValueError(f"{path} directory seed does not match checkpoint seed {seed}")
        profile_model, profile_train, _=load_chromatin_config(
            _resolve(root, main_profiles[profile_slug])
        )
        profile_train=replace(profile_train, seed=seed)
        if (
            state.get("model_config") != asdict(profile_model)
            or state.get("train_config") != asdict(profile_train)
        ):
            raise ValueError(f"{path} does not match its registered campaign profile")
        data_fingerprint=str(state.get("data_fingerprint", ""))
        if len(data_fingerprint) != 64 or any(
            char not in "0123456789abcdef" for char in data_fingerprint.lower()
        ):
            raise ValueError(f"{path} lacks a frozen data fingerprint")
        optimizer_step=int(state.get("optimizer_step", 0))
        validation=state.get("last_validation")
        best_validation_loss=state.get("best_validation_loss")
        if (
            optimizer_step < 1
            or not isinstance(validation, dict)
            or validation.get("schema") != "pdac-circuit.chromatin-validation/1"
            or int(validation.get("groups", 0)) < 3
            or validation.get("truncated") is not False
            or not isinstance(best_validation_loss, (int, float))
            or not math.isfinite(float(best_validation_loss))
            or not math.isclose(
                float(best_validation_loss),
                float(validation.get("group_mean_loss", float("nan"))),
                rel_tol=1e-9,
                abs_tol=1e-12,
            )
        ):
            raise ValueError(f"{path} is not a validation-selected non-truncated best checkpoint")
        if model_config is None:
            model_config=state.get("model_config")
            code_fingerprint=state.get("code_fingerprint")
        elif (
            state.get("model_config") != model_config
            or state.get("code_fingerprint") != code_fingerprint
        ):
            raise ValueError("final candidate seed checkpoints differ in model or code")
        records[seed]={
            "seed": seed,
            "checkpoint": str(path.relative_to(root)),
            "checkpoint_sha256": sha256_file(path),
            "config_hash": state.get("config_hash"),
            "code_fingerprint": state.get("code_fingerprint"),
            "data_fingerprint": data_fingerprint,
            "optimizer_step": optimizer_step,
            "best_validation_loss": best_validation_loss,
            "validation_groups": int(validation["groups"]),
        }
    if set(records) != set(required_seeds):
        raise ValueError(f"checkpoint seeds {sorted(records)} != required {required_seeds}")
    payload={
        "schema": RELEASE_SCHEMA,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "AUTHORIZED_AFTER_MODEL_FREEZE",
        "seal": str(Path(seal_report["path"]).relative_to(root)),
        "seal_sha256": seal_report["sha256"],
        "registered_seeds": required_seeds,
        "training_stage": "human_state_adaptation",
        "selected_campaign_profile": selected_profile,
        "selected_profile_config": main_profiles[selected_profile],
        "model_config": model_config,
        "code_fingerprint": code_fingerprint,
        "checkpoints": [records[seed] for seed in required_seeds],
        "protected_accessions": [
            row["accession"] for row in seal_report["seal"]["studies"]
        ],
        "test_tuning_uses_before_release": 0,
        "labels_may_open": True,
    }
    destination=_resolve(root, out)
    _atomic_json(destination, payload)
    return {
        "out": str(destination),
        "sha256": sha256_file(destination),
        "registered_seeds": required_seeds,
        "status": payload["status"],
    }

def validate_protected_metadata_release(
    project_root: str | Path,
    release_path: str | Path,
    *,
    accession: str,
) -> dict:
    root=Path(project_root).resolve()
    path=_resolve(root, release_path)
    if not path.is_file():
        return {"ok": False, "failures": [f"missing protected metadata release {path}"]}
    payload=json.loads(path.read_text(encoding="utf-8"))
    failures=[]
    if (
        payload.get("schema") != RELEASE_SCHEMA
        or payload.get("status") != "AUTHORIZED_AFTER_MODEL_FREEZE"
        or payload.get("labels_may_open") is not True
    ):
        failures.append("invalid protected metadata release schema/status")
    seal_path=_resolve(root, payload.get("seal", ""))
    seal_report=validate_protected_study_seal(root, seal_path, accession=accession)
    if not seal_report["ok"]:
        failures.extend(seal_report["failures"])
    elif payload.get("seal_sha256") != seal_report["sha256"]:
        failures.append("protected metadata release seal hash drifted")
    if accession.upper() not in payload.get("protected_accessions", []):
        failures.append(f"{accession.upper()} is absent from protected metadata release")
    if payload.get("registered_seeds") != seal_report.get("seal", {}).get(
        "registered_seeds"
    ):
        failures.append("protected metadata release seeds drifted")
    for row in payload.get("checkpoints", []):
        checkpoint=_resolve(root, row.get("checkpoint", ""))
        if not checkpoint.is_file() or row.get("checkpoint_sha256") != sha256_file(
            checkpoint
        ):
            failures.append(f"protected release checkpoint drifted for seed {row.get('seed')}")
    return {"ok": not failures, "failures": failures, "release": payload}
