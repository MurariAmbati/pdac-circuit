from __future__ import annotations

from datetime import date
import json
from pathlib import Path

from .streaming import sha256_file

SCHEMA = "pdac-circuit.chromatin-claim-surfaces/1"

def _resolve(root: Path,value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path

def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def _asset_contexts(asset: dict) -> list[str]:
    values: list[str] = []
    if asset.get("independence_context"):
        values.append(str(asset["independence_context"]))
    values.extend(str(value) for value in asset.get("independence_contexts",[]))
    return values

def validate_claim_surface_contract(
    project_root: str | Path,
    *,
    registry_path: str | Path = "chromatin_registry.json",
    assets_path: str | Path = "pdac_chromatin_assets.json",
    campaign_path: str | Path = "configs/chromatin-campaign.json",
    contract_path: str | Path = "configs/chromatin-claim-surfaces.json",
) -> dict:

    root = Path(project_root).resolve()
    registry_file = _resolve(root,registry_path)
    assets_file = _resolve(root,assets_path)
    campaign_file = _resolve(root,campaign_path)
    contract_file = _resolve(root,contract_path)
    registry = _load(registry_file)
    assets_payload = _load(assets_file)
    campaign = _load(campaign_file)
    contract = _load(contract_file)
    if contract.get("schema") != SCHEMA:
        raise ValueError("unsupported chromatin claim-surface contract schema")
    registered_path = registry.get("protected_study_policy",{}).get(
        "claim_surface_contract"
    )
    expected_relative = str(contract_file.relative_to(root)).replace("\\","/")
    if registered_path != expected_relative:
        raise ValueError("registry does not bind the claim-surface contract")
    amendment = registry.get("external_PDAC_perturbation_claim_amendment",{})
    if amendment.get("claim_surface_contract") != expected_relative:
        raise ValueError("external perturbation amendment does not bind the contract")
    cutoff = date.fromisoformat(str(contract["temporal_cutoff"]))
    if registry.get("split_policy",{}).get("temporal_cutoff") != cutoff.isoformat():
        raise ValueError("claim-surface temporal cutoff differs from the registry")
    if contract.get("target_signal_values_accessed_at_freeze") is not False:
        raise ValueError("claim surface was frozen after protected target access")
    if contract.get("failure_status") != "ABSTAIN":
        raise ValueError("claim-surface failure status must be ABSTAIN")

    assets = {str(row.get("id")): row for row in assets_payload.get("assets",[])}
    if len(assets) != len(assets_payload.get("assets",[])):
        raise ValueError("chromatin asset IDs are duplicated")
    rules = registry.get("rules",[])
    axes = contract.get("axes",[])
    if [row.get("axis") for row in axes] != [row.get("axis") for row in rules]:
        raise ValueError("claim-surface axes/order differ from benchmark rules")
    axis_by_name = {row["axis"]: row for row in axes}
    if len(axis_by_name) != len(axes):
        raise ValueError("claim-surface axes are duplicated")

    protected_policy = registry.get("protected_study_policy",{})
    policy_accessions = [str(value) for value in protected_policy.get("accessions",[])]
    policy_planes = protected_policy.get("planes",{})
    campaign_protected_planes = {
        str(row["name"]): [str(value) for value in row.get("sources",[])]
        for row in campaign.get("data_planes",[])
        if row.get("protected_from_training") is True
    }
    if campaign_protected_planes != policy_planes:
        raise ValueError("campaign protected planes differ from the registry policy")
    flattened = [
        accession
        for plane_accessions in policy_planes.values()
        for accession in plane_accessions
    ]
    if flattened != policy_accessions or len(flattened) != len(set(flattened)):
        raise ValueError("protected accessions are duplicated or ordered inconsistently")

    all_sources: set[str] = set()
    protected_axis_sources: set[str] = set()
    for rule,axis in zip(rules,axes,strict=True):
        for field in ("split","metric","minimum_groups"):
            if axis.get(field) != rule.get(field):
                raise ValueError(
                    f"{axis['axis']} {field} differs between rule and source contract"
                )
        sources = [str(value) for value in axis.get("source_accessions",[])]
        if not sources or len(sources) != len(set(sources)):
            raise ValueError(f"{axis['axis']} sources are missing or duplicated")
        missing = sorted(set(sources) - set(assets))
        if missing:
            raise ValueError(f"{axis['axis']} uses unregistered assets: {missing}")
        all_sources.update(sources)
        for accession in sources:
            released = assets[accession].get("release_date")
            if released and date.fromisoformat(str(released)) > cutoff:
                raise ValueError(f"{axis['axis']} source {accession} is post-cutoff")
        if axis.get("protected_target") is True:
            plane = str(axis.get("protected_plane",""))
            if sources != policy_planes.get(plane):
                raise ValueError(
                    f"{axis['axis']} sources differ from protected plane {plane!r}"
                )
            protected_axis_sources.update(sources)
        elif set(sources) & set(policy_accessions):
            raise ValueError(f"open axis {axis['axis']} includes a protected source")
        if not str(axis.get("independence_unit","")).strip():
            raise ValueError(f"{axis['axis']} lacks an independence unit")
        if not str(axis.get("leakage_barrier","")).strip():
            raise ValueError(f"{axis['axis']} lacks a leakage barrier")
        if not str(axis.get("target_construction","")).strip():
            raise ValueError(f"{axis['axis']} lacks a target construction")
        if not str(axis.get("status","")).strip():
            raise ValueError(f"{axis['axis']} lacks a build status")

    if protected_axis_sources != set(policy_accessions):
        raise ValueError("not every protected accession belongs to a protected claim axis")
    if "GSE146486" in all_sources:
        raise ValueError("pancreatic-development GSE146486 leaked into a PDAC claim axis")
    rejected = assets.get("GSE146486",{})
    if rejected.get("split") != "excluded_from_PDAC_claim":
        raise ValueError("GSE146486 is not explicitly excluded from the PDAC claim")

    external_axis = axis_by_name.get("external_KLF5_perturbation_direction",{})
    expected_external_sources = amendment.get("protected_external_perturbation_studies")
    if external_axis.get("source_accessions") != expected_external_sources:
        raise ValueError("external KLF5 studies drifted from the registry amendment")
    allowed_groups = [str(value) for value in external_axis.get("allowed_groups",[])]
    if allowed_groups != amendment.get("registered_external_contexts"):
        raise ValueError("external KLF5 groups drifted from the registry amendment")
    if (
        len(allowed_groups) != external_axis.get("minimum_groups")
        or len(allowed_groups) != len(set(allowed_groups))
    ):
        raise ValueError("external KLF5 independent groups are infeasible or duplicated")
    asset_contexts = {
        context
        for accession in external_axis["source_accessions"]
        for context in _asset_contexts(assets[accession])
    }
    if asset_contexts != set(allowed_groups):
        raise ValueError("external KLF5 asset contexts differ from allowed groups")
    external_rule = next(
        row
        for row in rules
        if row.get("axis") == "external_KLF5_perturbation_direction"
    )
    if (
        external_rule.get("allowed_groups") != allowed_groups
        or external_rule.get("exact_groups_required") is not True
    ):
        raise ValueError("external KLF5 benchmark does not enforce exact groups")

    within_axis = axis_by_name.get("within_study_perturbation_direction",{})
    within_rule = next(
        row
        for row in rules
        if row.get("axis") == "within_study_perturbation_direction"
    )
    within_groups = [str(value) for value in within_axis.get("allowed_groups",[])]
    if (
        within_axis.get("source_accessions") != ["GSE99311"]
        or within_axis.get("primary_assays") != ["H3K27ac"]
        or within_axis.get("secondary_diagnostics") != ["FOXA1 occupancy direction"]
        or within_axis.get("minimum_coordinate_overlap_fraction") != 0.995
        or within_axis.get("exact_groups_required") is not True
        or len(within_groups) != within_axis.get("minimum_groups")
        or len(within_groups) != len(set(within_groups))
    ):
        raise ValueError("within-study GSE99311 assay or independence contract drifted")
    if (
        within_rule.get("primary_assays") != within_axis.get("primary_assays")
        or within_rule.get("minimum_coordinate_overlap_fraction")
        != within_axis.get("minimum_coordinate_overlap_fraction")
        or within_rule.get("allowed_groups") != within_groups
        or within_rule.get("exact_groups_required") is not True
    ):
        raise ValueError("within-study benchmark does not enforce exact biological contexts")

    stage_data = campaign.get("execution",{}).get("stage_data",{})
    serialized_stage_data = json.dumps(stage_data).lower()
    leaked = [
        accession
        for accession in policy_accessions
        if accession.lower() in serialized_stage_data
    ]
    if leaked:
        raise ValueError(f"protected claim sources leaked into training inputs: {leaked}")
    return {
        "ok": True,
        "schema": SCHEMA,
        "contract": expected_relative,
        "sha256": sha256_file(contract_file),
        "axes": len(axes),
        "required_axes": sum(bool(row.get("required_for_claim")) for row in rules),
        "registered_sources": len(all_sources),
        "protected_accessions": policy_accessions,
        "external_perturbation_groups": allowed_groups,
        "within_study_perturbation_groups": within_groups,
    }
