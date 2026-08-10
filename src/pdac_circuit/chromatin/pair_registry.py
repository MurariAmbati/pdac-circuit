from __future__ import annotations

import json
from pathlib import Path
import re
import uuid

from .pairing import compose_paired_shards
from .streaming import sha256_file

def build_external_perturbation_pair_plan(
    project_root: str | Path,
    track_index_paths: list[str | Path],
    *,
    contract_path: str | Path = "configs/chromatin-claim-surfaces.json",
    out: str | Path = "data/pair_specs/external_KLF5.intervention.json",
    merged_index_out: str | Path = "data/evaluation_track_specs/external_KLF5/index.json",
) -> dict:

    from .claim_surfaces import validate_claim_surface_contract

    root = Path(project_root).resolve()
    contract_file = Path(contract_path)
    if not contract_file.is_absolute():
        contract_file = root / contract_file
    contract_report = validate_claim_surface_contract(
        root,contract_path=contract_file
    )
    contract = json.loads(contract_file.read_text(encoding="utf-8"))
    axis = next(
        row
        for row in contract["axes"]
        if row["axis"] == "external_KLF5_perturbation_direction"
    )
    expected_studies = list(axis["source_accessions"])
    allowed_groups = list(axis["allowed_groups"])
    expected_assays = {
        group: list(values)
        for group,values in axis["primary_assays_by_group"].items()
    }
    minimum_replicates = int(axis["minimum_matched_replicates_per_assay"])
    if minimum_replicates < 2:
        raise ValueError("external perturbation requires at least two matched replicates")
    index_paths = []
    indexes = []
    for value in track_index_paths:
        path = Path(value)
        if not path.is_absolute():
            path = root / path
        index = json.loads(path.read_text(encoding="utf-8"))
        if (
            index.get("schema") != "pdac-circuit.geo-track-specs/1"
            or index.get("evaluation_only") is not True
            or index.get("failures")
        ):
            raise ValueError(f"external evaluation TrackSpec index is invalid: {path}")
        index_paths.append(path)
        indexes.append(index)
    observed_studies = [str(index.get("accession")) for index in indexes]
    if observed_studies != expected_studies:
        raise ValueError(
            f"external indexes {observed_studies} differ from {expected_studies}"
        )

    selected_rows: dict[str,dict] = {}
    groups: dict[str,dict[str,list[dict]]] = {}
    for index in indexes:
        for row in index.get("written",[]):
            spec_path = root / row["spec"]
            spec = json.loads(spec_path.read_text(encoding="utf-8"))
            group = str(spec.get("sample_group",""))
            assay = str(row.get("assay",""))
            if group not in allowed_groups or assay not in expected_assays[group]:
                continue
            relation = str(spec.get("pair_relation",""))
            perturbation = [float(value) for value in spec.get("perturbation_features",[])]
            if len(perturbation) != 22:
                raise ValueError(f"{spec_path} has an invalid perturbation vector")
            primary = relation == "control" and not any(perturbation)
            primary = primary or (
                relation == "intervention"
                and perturbation[16] == -1.0
                and abs(perturbation[19] - (4.0 / 24.0)) < 5e-4
            )
            if not primary:
                continue
            pair_group = str(spec.get("pair_group",""))
            if not pair_group:
                raise ValueError(f"primary external track lacks pair_group: {spec_path}")
            record = {
                "track": spec["accession"],
                "spec": str(spec_path.relative_to(root)).replace("\\","/"),
                "spec_sha256": sha256_file(spec_path),
                "gsm": spec.get("sample_accession",""),
                "genome": spec.get("genome",""),
                "perturbation_label": spec.get("perturbation_label",""),
                "pair_control_family": spec.get("pair_control_family",""),
                "independence_group": group,
                "assay": assay,
                "biological_replicate": spec.get("biological_replicate",""),
                "study": spec.get("study",""),
            }
            groups.setdefault(pair_group,{"control": [],"intervention": []})[
                relation
            ].append(record)
            selected_rows[str(spec["accession"])] = row

    pairs,unresolved = [],[]
    observed_group_assays: dict[str,dict[str,int]] = {
        group: {} for group in allowed_groups
    }
    for pair_group,members in sorted(groups.items()):
        controls = members["control"]
        interventions = members["intervention"]
        if len(controls) != 1 or len(interventions) != 1:
            unresolved.append(
                {
                    "pair_group": pair_group,
                    "reason": "primary external pair requires exactly one 0h and one 4h track",
                    "controls": controls,
                    "interventions": interventions,
                }
            )
            continue
        control,treatment = controls[0],interventions[0]
        comparable = (
            control["pair_control_family"] == treatment["pair_control_family"]
            and control["genome"] == treatment["genome"]
            and control["independence_group"] == treatment["independence_group"]
            and control["assay"] == treatment["assay"]
            and control["biological_replicate"] == treatment["biological_replicate"]
        )
        if not comparable:
            unresolved.append(
                {
                    "pair_group": pair_group,
                    "reason": "0h/4h technology, genome, context, assay, or replicate mismatch",
                    "control": control,
                    "intervention": treatment,
                }
            )
            continue
        group = treatment["independence_group"]
        assay = treatment["assay"]
        observed_group_assays[group][assay] = observed_group_assays[group].get(assay,0) + 1
        pairs.append(
            {
                "pair_id": f"{control['track']}__TO__{treatment['track']}",
                "pair_group": pair_group,
                "mode": "perturbation",
                "independence_group": group,
                "assay": assay,
                "primary_contrast": "4h_minus_0h",
                "reference": control,
                "treatment": treatment,
                "status": "registered_external_primary_contrast",
            }
        )
    for group in allowed_groups:
        if set(observed_group_assays[group]) != set(expected_assays[group]):
            unresolved.append(
                {
                    "independence_group": group,
                    "reason": "registered primary assay set is incomplete",
                    "expected": expected_assays[group],
                    "observed": sorted(observed_group_assays[group]),
                }
            )
        for assay in expected_assays[group]:
            count = observed_group_assays[group].get(assay,0)
            if count < minimum_replicates:
                unresolved.append(
                    {
                        "independence_group": group,
                        "assay": assay,
                        "reason": (
                            f"only {count} matched replicates; require {minimum_replicates}"
                        ),
                    }
                )

    merged_index_path = Path(merged_index_out)
    if not merged_index_path.is_absolute():
        merged_index_path = root / merged_index_path
    merged_index_path.parent.mkdir(parents=True,exist_ok=True)
    merged_index = {
        "schema": "pdac-circuit.geo-track-specs/1",
        "accession": "EXTERNAL_KLF5",
        "evaluation_only": True,
        "claim_surface_contract": contract_report["contract"],
        "claim_surface_contract_sha256": contract_report["sha256"],
        "source_track_indexes": [
            {
                "path": str(path.relative_to(root)).replace("\\","/"),
                "sha256": sha256_file(path),
            }
            for path in index_paths
        ],
        "written": [selected_rows[key] for key in sorted(selected_rows)],
        "failures": unresolved,
        "excluded_profiles": [],
        "state_policy": "Protected external evaluation only; never a training index.",
    }
    merged_index_path.write_text(
        json.dumps(merged_index,indent=2,sort_keys=True) + "\n",encoding="utf-8"
    )
    plan_path = Path(out)
    if not plan_path.is_absolute():
        plan_path = root / plan_path
    plan_path.parent.mkdir(parents=True,exist_ok=True)
    report = {
        "schema": "pdac-circuit.intervention-pair-plan/1",
        "study": "EXTERNAL_KLF5",
        "axis": "external_KLF5_perturbation_direction",
        "track_index": str(merged_index_path.relative_to(root)).replace("\\","/"),
        "track_index_sha256": sha256_file(merged_index_path),
        "claim_surface_contract": contract_report["contract"],
        "claim_surface_contract_sha256": contract_report["sha256"],
        "allowed_groups": allowed_groups,
        "observed_group_assays": observed_group_assays,
        "pairs": pairs,
        "unresolved": unresolved,
        "state_pair_policy": (
            "Exactly 4h KLF5 dTAG minus matched 0h within cell line, assay, replicate, "
            "technology, and genome. Replicates and assays stay nested within three contexts."
        ),
    }
    plan_path.write_text(
        json.dumps(report,indent=2,sort_keys=True) + "\n",encoding="utf-8"
    )
    return {
        **report,
        "plan_path": str(plan_path),
        "merged_index_path": str(merged_index_path),
    }

def build_intervention_pair_plan(
    project_root: str | Path,
    track_index_path: str | Path,
    *,
    out: str | Path | None = None,
) -> dict:

    root = Path(project_root)
    index_path = Path(track_index_path)
    index = json.loads(index_path.read_text(encoding="utf-8"))
    if index.get("schema") != "pdac-circuit.geo-track-specs/1":
        raise ValueError("pair planning requires a GEO TrackSpec index")
    groups: dict[str,dict[str,list[dict]]] = {}
    for row in index.get("written",[]):
        spec_path = root / row["spec"]
        spec = json.loads(spec_path.read_text(encoding="utf-8"))
        pair_group = str(spec.get("pair_group",""))
        relation = str(spec.get("pair_relation",""))
        if not pair_group or relation not in {"control","intervention"}:
            continue
        groups.setdefault(pair_group,{"control": [],"intervention": []})[relation].append(
            {
                "track": spec["accession"],
                "spec": str(spec_path),
                "spec_sha256": sha256_file(spec_path),
                "gsm": spec.get("sample_accession",""),
                "genome": spec.get("genome",""),
                "perturbation_label": spec.get("perturbation_label",""),
                "pair_control_family": spec.get("pair_control_family",""),
                "assay": row.get("assay",""),
                "independence_group": spec.get("sample_group",""),
            }
        )

    pairs,unresolved = [],[]
    for pair_group,members in sorted(groups.items()):
        controls = members["control"]
        interventions = members["intervention"]
        if not interventions:
            continue
        if len(controls) != 1:
            unresolved.append(
                {
                    "pair_group": pair_group,
                    "reason": f"requires exactly one control, found {len(controls)}",
                    "controls": controls,
                    "interventions": interventions,
                }
            )
            continue
        control = controls[0]
        for treatment in interventions:
            if (
                not control["pair_control_family"]
                or not treatment["pair_control_family"]
                or control["pair_control_family"] != treatment["pair_control_family"]
                or treatment["pair_control_family"] == "unperturbed"
            ):
                unresolved.append(
                    {
                        "pair_group": pair_group,
                        "reason": "control/intervention technology family is absent or mismatched",
                        "control": control,
                        "intervention": treatment,
                    }
                )
                continue
            if control["genome"] != treatment["genome"]:
                unresolved.append(
                    {
                        "pair_group": pair_group,
                        "reason": "control/intervention genome mismatch",
                        "control": control,
                        "intervention": treatment,
                    }
                )
                continue
            if (
                not control["assay"]
                or control["assay"] != treatment["assay"]
                or not control["independence_group"]
                or control["independence_group"] != treatment["independence_group"]
            ):
                unresolved.append(
                    {
                        "pair_group": pair_group,
                        "reason": "control/intervention assay or independence context is absent or mismatched",
                        "control": control,
                        "intervention": treatment,
                    }
                )
                continue
            pairs.append(
                {
                    "pair_id": f"{control['track']}__TO__{treatment['track']}",
                    "pair_group": pair_group,
                    "mode": "perturbation",
                    "assay": treatment["assay"],
                    "independence_group": treatment["independence_group"],
                    "reference": control,
                    "treatment": treatment,
                    "status": "registered_from_depositor_metadata",
                }
            )
    report = {
        "schema": "pdac-circuit.intervention-pair-plan/1",
        "study": index.get("accession"),
        "track_index": str(index_path),
        "track_index_sha256": sha256_file(index_path),
        "pairs": pairs,
        "unresolved": unresolved,
        "state_pair_policy": (
            "No state pairs generated. Normal, PanIN, primary, and metastatic organoid lines "
            "remain independent groups unless depositor metadata proves a matched lineage."
        ),
    }
    destination = Path(out) if out else (
        root / "data" / "pair_specs" / f"{index.get('accession')}.intervention.json"
    )
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.write_text(json.dumps(report,indent=2,sort_keys=True),encoding="utf-8")
    report["plan_path"] = str(destination)
    return report

def _pair_directory_name(pair_id: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+","_",pair_id).strip("._")
    if not safe:
        raise ValueError("pair ID cannot be converted to a safe directory name")
    if len(safe) <= 96:
        return safe
    import hashlib

    return f"{safe[:75]}-{hashlib.sha256(pair_id.encode('utf-8')).hexdigest()[:16]}"

def verify_paired_output(path: str | Path,*,expected_pair_id: str) -> dict:
    destination = Path(path)
    manifest_path = destination / "manifest.json"
    failures = []
    if not manifest_path.is_file():
        return {"valid": False,"failures": ["missing manifest.json"]}
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError,json.JSONDecodeError) as exc:
        return {"valid": False,"failures": [f"invalid manifest: {exc}"]}
    if manifest.get("schema") != "pdac-circuit.paired-chromatin-shards/1":
        failures.append("unsupported paired manifest schema")
    if manifest.get("pair_id") != expected_pair_id:
        failures.append("pair ID mismatch")
    if manifest.get("mode") != "perturbation" or manifest.get("valid") is not True:
        failures.append("paired manifest is not a valid perturbation composition")
    for role in ("reference","treatment"):
        for row in manifest.get(f"{role}_shards",[]):
            source = Path(str(row.get("path","")))
            if not source.is_file() or sha256_file(source) != row.get("sha256"):
                failures.append(f"{role} source shard drift: {source}")
    shard_examples = shard_bytes = 0
    for row in manifest.get("shards",[]):
        shard = destination / str(row.get("path",""))
        if (
            not shard.is_file()
            or shard.stat().st_size != row.get("bytes")
            or sha256_file(shard) != row.get("sha256")
        ):
            failures.append(f"paired output shard drift: {shard}")
            continue
        shard_examples += int(row.get("examples",0))
        shard_bytes += shard.stat().st_size
    if not manifest.get("shards") or shard_examples != manifest.get("matched_examples"):
        failures.append("paired shard/example accounting mismatch")
    return {
        "valid": not failures,
        "failures": failures,
        "manifest": manifest,
        "manifest_path": str(manifest_path),
        "manifest_sha256": sha256_file(manifest_path),
        "examples": shard_examples,
        "shards": len(manifest.get("shards",[])),
        "bytes": shard_bytes,
        "source_shards_verified": not any("source shard drift" in row for row in failures),
    }

def materialize_intervention_pair_plan(
    project_root: str | Path,
    pair_plan_path: str | Path,
    compiled_root: str | Path,
    output_root: str | Path,
    *,
    windows_per_shard: int = 64,
    minimum_overlap_fraction: float = 0.80,
) -> dict:

    root = Path(project_root).resolve()
    plan_path = Path(pair_plan_path)
    if not plan_path.is_absolute():
        plan_path = root / plan_path
    source_root = Path(compiled_root)
    if not source_root.is_absolute():
        source_root = root / source_root
    destination = Path(output_root)
    if not destination.is_absolute():
        destination = root / destination
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("schema") != "pdac-circuit.intervention-pair-plan/1":
        raise ValueError("materialization requires an intervention pair plan")
    if plan.get("unresolved"):
        raise RuntimeError(
            f"pair plan has {len(plan['unresolved'])} unresolved groups; refusing partial pairing"
        )
    pairs = plan.get("pairs",[])
    if not pairs:
        raise RuntimeError("pair plan contains no registered interventions")
    source_completion_path = source_root / "_COMPLETE.json"
    source_completion = json.loads(source_completion_path.read_text(encoding="utf-8"))
    if (
        source_completion.get("schema")
        != "pdac-circuit.compiled-collection-completion/1"
        or source_completion.get("successful") is not True
        or source_completion.get("track_index_sha256") != plan.get("track_index_sha256")
    ):
        raise RuntimeError("compiled source collection is incomplete or not bound to pair plan")
    destination.mkdir(parents=True,exist_ok=True)
    verified = []
    for pair in sorted(pairs,key=lambda row: row["pair_id"]):
        pair_id = str(pair["pair_id"])
        reference_accession = str(pair["reference"]["track"])
        treatment_accession = str(pair["treatment"]["track"])
        reference = sorted((source_root / reference_accession).glob("*.npz"))
        treatment = sorted((source_root / treatment_accession).glob("*.npz"))
        if not reference or not treatment:
            raise FileNotFoundError(
                f"compiled shards missing for registered pair {pair_id}"
            )
        pair_destination = destination / _pair_directory_name(pair_id)
        if pair_destination.exists():
            audit = verify_paired_output(
                pair_destination,expected_pair_id=pair_id
            )
            if not audit["valid"]:
                raise RuntimeError(
                    f"existing paired output {pair_destination} is invalid: "
                    f"{audit['failures'][:3]}"
                )
        else:
            compose_paired_shards(
                reference,
                treatment,
                pair_destination,
                mode="perturbation",
                windows_per_shard=windows_per_shard,
                minimum_overlap_fraction=minimum_overlap_fraction,
                pair_id=pair_id,
            )
            audit = verify_paired_output(
                pair_destination,expected_pair_id=pair_id
            )
            if not audit["valid"]:
                raise RuntimeError(
                    f"new paired output {pair_destination} failed verification: "
                    f"{audit['failures'][:3]}"
                )
        manifest = audit["manifest"]
        overlap_fraction = float(manifest.get("overlap_fraction",0.0))
        if overlap_fraction < minimum_overlap_fraction:
            raise RuntimeError(
                f"paired output {pair_destination} overlap {overlap_fraction:.6f} "
                f"is below collection floor {minimum_overlap_fraction:.6f}"
            )
        verified.append(
            {
                "pair_id": pair_id,
                "pair_group": pair.get("pair_group"),
                "independence_group": pair.get("independence_group"),
                "assay": pair.get("assay"),
                "output": str(pair_destination.relative_to(root)),
                "manifest_sha256": audit["manifest_sha256"],
                "examples": audit["examples"],
                "shards": audit["shards"],
                "bytes": audit["bytes"],
                "sequence_length": manifest["sequence_length"],
                "bin_size": manifest["bin_size"],
                "negative_keep_probability": manifest[
                    "negative_keep_probability"
                ],
                "overlap_fraction": overlap_fraction,
                "composition_minimum_overlap_fraction": float(
                    manifest.get("minimum_overlap_fraction",0.0)
                ),
                "source_shards_verified": audit["source_shards_verified"],
                "valid": True,
            }
        )
    contracts = {
        (
            row["sequence_length"],
            row["bin_size"],
            row["negative_keep_probability"],
        )
        for row in verified
    }
    if len(contracts) != 1:
        raise RuntimeError(f"paired outputs mix compile contracts: {sorted(contracts)}")
    sequence_length,bin_size,negative_keep_probability = next(iter(contracts))
    completion = {
        "schema": "pdac-circuit.paired-collection-completion/1",
        "successful": True,
        "pair_plan": str(plan_path.relative_to(root)),
        "pair_plan_sha256": sha256_file(plan_path),
        "source_completion": str(source_completion_path.relative_to(root)),
        "source_completion_sha256": sha256_file(source_completion_path),
        "sequence_length": sequence_length,
        "bin_size": bin_size,
        "negative_keep_probability": negative_keep_probability,
        "minimum_required_overlap_fraction": float(minimum_overlap_fraction),
        "minimum_observed_overlap_fraction": min(
            row["overlap_fraction"] for row in verified
        ),
        "registered_pairs": len(pairs),
        "verified_pairs": len(verified),
        "examples": sum(row["examples"] for row in verified),
        "shards": sum(row["shards"] for row in verified),
        "bytes": sum(row["bytes"] for row in verified),
        "pairs": verified,
    }
    completion_path = destination / "_COMPLETE.json"
    temporary = destination / f"._COMPLETE.partial-{uuid.uuid4().hex}.json"
    temporary.write_text(
        json.dumps(completion,indent=2,sort_keys=True) + "\n",encoding="utf-8"
    )
    temporary.replace(completion_path)
    return {**completion,"completion_marker": str(completion_path)}
