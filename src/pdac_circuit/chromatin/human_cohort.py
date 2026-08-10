from __future__ import annotations

from datetime import date
import json
from pathlib import Path

SCHEMA="pdac-circuit.human-adaptation-cohort/1"

def _load(path: Path) -> dict:
    payload=json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload

def _validate_available_track_index(
    root: Path, accession: str, role: str
) -> tuple[str | None, int]:

    index_path=root / "data" / "track_specs" / accession / "index.json"
    if not index_path.is_file():
        return str(index_path.relative_to(root)).replace("\\", "/"), 0
    index=_load(index_path)
    if index.get("schema") != "pdac-circuit.geo-track-specs/1":
        raise ValueError(f"{accession} has an unsupported TrackSpec index")
    if index.get("accession") != accession or index.get("failures"):
        raise ValueError(f"{accession} TrackSpec index is incomplete or mismatched")
    rows=index.get("written", [])
    if not rows:
        raise ValueError(f"{accession} TrackSpec index contains no eligible profiles")
    for row in rows:
        spec_path=root / str(row.get("spec") or "")
        if not spec_path.is_file():
            raise ValueError(f"{accession} TrackSpec is missing: {spec_path}")
        spec=_load(spec_path)
        split_role=spec.get("split_role")
        if row.get("split_role") != split_role or spec.get("study") != accession:
            raise ValueError(f"{accession} TrackSpec identity drifted from its index")
        if role == "validation_only" and split_role != "validation_study":
            raise ValueError(f"{accession} validation profile can contribute gradients")
        if role == "training" and split_role in {
            "validation_study",
            "external_study",
            "temporal_test",
        }:
            raise ValueError(f"{accession} training profile has a protected split role")
    return None, len(rows)

def validate_human_cohort_contract(
    project_root: str | Path,
    contract_path: str | Path = "configs/chromatin-human-cohort.json",
    *,
    require_study_plans: bool = False,
    require_track_indexes: bool = False,
) -> dict:
    root=Path(project_root).resolve()
    path=Path(contract_path)
    if not path.is_absolute():
        path=root / path
    contract=_load(path)
    if contract.get("schema") != SCHEMA:
        raise ValueError("unsupported human adaptation cohort schema")
    cutoff=date.fromisoformat(contract["temporal_cutoff"])

    assets_payload=_load(root / "pdac_chromatin_assets.json")
    assets={row["id"]: row for row in assets_payload.get("assets", [])}
    registry=_load(root / "chromatin_registry.json")
    campaign=_load(root / "configs" / "chromatin-campaign.json")
    amendment=registry.get("human_adaptation_cohort_amendment", {})
    if amendment.get("cohort_contract") != str(path.relative_to(root)).replace("\\", "/"):
        raise ValueError("registry does not bind the human adaptation cohort contract")
    if registry.get("split_policy", {}).get("temporal_cutoff") != contract["temporal_cutoff"]:
        raise ValueError("cohort temporal cutoff differs from the registry")

    planes=contract.get("planes", [])
    accessions=[row.get("accession") for row in planes]
    if not accessions or len(accessions) != len(set(accessions)):
        raise ValueError("cohort accessions must be present and unique")
    by_accession={row["accession"]: row for row in planes}
    role_policy=contract.get("role_policy", {})
    role_members=[item for members in role_policy.values() for item in members]
    if len(role_members) != len(set(role_members)) or set(role_members) != set(accessions):
        raise ValueError("every cohort plane must have exactly one registered role")
    for role, members in role_policy.items():
        for accession in members:
            if by_accession[accession].get("role") != role:
                raise ValueError(f"{accession} role differs between plane and role policy")

    training=role_policy.get("training", [])
    validation=role_policy.get("validation_only", [])
    protected=role_policy.get("protected_test", [])
    excluded=role_policy.get("profile_training_excluded", [])
    if training != amendment.get("registered_open_training_studies"):
        raise ValueError("training studies drifted from the frozen registry amendment")
    if validation != [amendment.get("registered_validation_study")]:
        raise ValueError("validation study drifted from the frozen registry amendment")
    if protected != amendment.get("protected_test_studies"):
        raise ValueError("protected test studies drifted from the frozen registry amendment")
    protected_policy=registry.get("protected_study_policy", {})
    if protected != protected_policy.get("planes", {}).get("human_external"):
        raise ValueError("protected patient-test studies drifted from the release gate")

    missing_assets=[accession for accession in accessions if accession not in assets]
    if missing_assets:
        raise ValueError(f"cohort assets are unregistered: {missing_assets}")
    for accession in training + validation:
        released=assets[accession].get("release_date")
        if not released or date.fromisoformat(released) > cutoff:
            raise ValueError(f"{accession} is not an eligible pre-cutoff open study")
    validation_accession=validation[0]
    validation_plane=by_accession[validation_accession]
    if assets[validation_accession].get("split") != "validation_study_by_patient":
        raise ValueError("patient validation study is not frozen validation-only")
    if validation_plane.get("gradient_updates") is not False:
        raise ValueError("patient validation study permits gradient updates")
    if validation_plane.get("architecture_or_threshold_tuning") is not False:
        raise ValueError("patient validation study permits architecture or threshold tuning")
    forbidden_outcomes=validation_plane.get("forbidden_outcome_characteristics", [])
    if not forbidden_outcomes or forbidden_outcomes != assets[validation_accession].get(
        "forbidden_outcome_characteristics"
    ):
        raise ValueError("patient outcome-redaction fields drifted from the asset registry")
    for accession in protected:
        if "test" not in str(assets[accession].get("split", "")):
            raise ValueError(f"{accession} is not protected as a test study")

    expected_globs=[
        by_accession[accession]["compiled_glob"] for accession in training + validation
    ]
    stage_globs=campaign.get("execution", {}).get("stage_data", {}).get(
        "human_state_adaptation"
    )
    if stage_globs != expected_globs:
        raise ValueError("campaign human-adaptation inputs drifted from the cohort contract")
    serialized_globs=json.dumps(stage_globs)
    if any(accession.lower() in serialized_globs.lower() for accession in protected + excluded):
        raise ValueError("protected or excluded studies leaked into profile training")

    missing_track_indexes=[]
    indexed_profiles={}
    for accession in training + validation:
        missing, profiles=_validate_available_track_index(
            root, accession, by_accession[accession]["role"]
        )
        if missing:
            missing_track_indexes.append(accession)
        else:
            indexed_profiles[accession]=profiles
    if require_track_indexes and missing_track_indexes:
        raise FileNotFoundError(
            f"GEO TrackSpec indexes are missing: {missing_track_indexes}"
        )

    missing_plans=[
        accession
        for accession in training + validation + excluded
        if not (root / "data" / "manifests" / "studies" / f"{accession}.plan.json").is_file()
    ]
    if require_study_plans and missing_plans:
        raise FileNotFoundError(f"frozen GEO plans are missing: {missing_plans}")
    return {
        "schema": SCHEMA,
        "contract": str(path.relative_to(root)).replace("\\", "/"),
        "training_studies": training,
        "validation_studies": validation,
        "protected_test_studies": protected,
        "profile_training_excluded": excluded,
        "stage_globs": stage_globs,
        "missing_plans": missing_plans,
        "missing_track_indexes": missing_track_indexes,
        "indexed_profiles": indexed_profiles,
        "ok": True,
    }
