from __future__ import annotations

from datetime import datetime
import glob
import hashlib
import json
from pathlib import Path
import shlex
import uuid
from collections.abc import Sequence

import numpy as np

TRAINING_STAGES=(
    "healthy_prior",
    "progression_state_residual",
    "signed_intervention_residual",
    "human_state_adaptation",
)

def _json_hash(payload: dict) -> str:
    encoded=json.dumps(payload,sort_keys=True,separators=(",",":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

def _collection_root(root: Path,pattern: str) -> Path:
    wildcard_positions=[pattern.find(char) for char in "*[?" if char in pattern]
    prefix=pattern[: min(wildcard_positions)] if wildcard_positions else pattern
    return (root / prefix.rstrip("/\\")).resolve()

def _glob_summary(
    root: Path,
    pattern: str,
    expected_sequence_length: int,
    expected_bin_size: int,
    expected_negative_keep_probability: float,
    *,
    require_completion_marker: bool,
    allow_local_tiling: bool,
    expected_conditioning_dimensions: dict[str,int] | None = None,
) -> dict:
    absolute_pattern=str(root / pattern)
    paths=sorted(
        path.resolve()
        for path in (Path(value) for value in glob.glob(absolute_pattern,recursive=True))
        if path.suffix.lower() == ".npz" and path.is_file()
    )
    manifest_lengths=set()
    negative_keep_probabilities=set()
    conditioning_lengths: dict[str,set[int]] = {
        key: set() for key in (expected_conditioning_dimensions or {})
    }
    conditioning_compatible=True
    zero_perturbation_padding_manifests=0
    missing_manifests=0
    for parent in {path.parent for path in paths}:
        manifest_path=parent / "manifest.json"
        if not manifest_path.is_file():
            missing_manifests += 1
            continue
        manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_lengths.add(manifest.get("sequence_length"))
        negative_keep_probabilities.add(manifest.get("negative_keep_probability"))
        track=manifest.get("track",{})
        for key,expected in (expected_conditioning_dimensions or {}).items():
            values=track.get(key)
            if (
                not isinstance(values,list)
                and manifest.get("schema")
                == "pdac-circuit.paired-chromatin-shards/1"
            ):
                dimension=manifest.get("conditioning_dimensions",{}).get(key)
                if not isinstance(dimension,int):
                    shard_rows=manifest.get("shards",[])
                    shard_path=(
                        parent / str(shard_rows[0].get("path",""))
                        if shard_rows
                        else None
                    )
                    if shard_path is not None and shard_path.is_file():
                        with np.load(shard_path,allow_pickle=False) as shard:
                            array=shard[key] if key in shard.files else None
                            if array is not None and array.ndim == 2:
                                dimension=int(array.shape[1])
                if isinstance(dimension,int):
                    conditioning_lengths[key].add(dimension)
                    if dimension != expected:
                        conditioning_compatible=False
                    continue
            if not isinstance(values,list):
                conditioning_compatible=False
                continue
            conditioning_lengths[key].add(len(values))
            if len(values) == expected:
                continue
            if (
                key == "perturbation_features"
                and 0 < len(values) < expected
                and not any(values)
            ):
                zero_perturbation_padding_manifests += 1
                continue
            conditioning_compatible=False
    collection_root=_collection_root(root,pattern)
    completion_path=collection_root / "_COMPLETE.json"
    completion=None
    completion_valid=not require_completion_marker
    completion_index_valid=not require_completion_marker
    if completion_path.is_file():
        try:
            completion=json.loads(completion_path.read_text(encoding="utf-8"))
            schema=completion.get("schema")
            geometry_valid=(
                completion.get("sequence_length") == expected_sequence_length
                or (
                    allow_local_tiling
                    and isinstance(completion.get("sequence_length"),int)
                    and completion["sequence_length"] > expected_sequence_length
                    and completion["sequence_length"] % expected_sequence_length == 0
                )
            ) and completion.get("bin_size") == expected_bin_size
            if schema == "pdac-circuit.compiled-collection-completion/1":
                index_path=Path(str(completion.get("track_index") or ""))
                if not index_path.is_absolute():
                    index_path=root / index_path
                index_payload=json.loads(index_path.read_text(encoding="utf-8"))
                completion_index_valid=(
                    not index_payload.get("failures")
                    and len(index_payload.get("written",[]))
                    == completion.get("registered_tracks")
                    and hashlib.sha256(index_path.read_bytes()).hexdigest()
                    == completion.get("track_index_sha256")
                )
                completion_valid=(
                    completion.get("successful") is True
                    and geometry_valid
                    and completion.get("registered_tracks")
                    == completion.get("verified_tracks")
                    and completion.get("registered_tracks",0) > 0
                    and len(completion.get("tracks",[]))
                    == completion.get("registered_tracks")
                    and all(
                        row.get("source_sha256_verified") is True
                        for row in completion.get("tracks",[])
                    )
                    and completion_index_valid
                )
            elif schema == "pdac-circuit.paired-collection-completion/1":
                pair_plan_path=Path(str(completion.get("pair_plan") or ""))
                source_completion_path=Path(
                    str(completion.get("source_completion") or "")
                )
                if not pair_plan_path.is_absolute():
                    pair_plan_path=root / pair_plan_path
                if not source_completion_path.is_absolute():
                    source_completion_path=root / source_completion_path
                pair_plan=json.loads(pair_plan_path.read_text(encoding="utf-8"))
                completion_index_valid=(
                    pair_plan.get("schema")
                    == "pdac-circuit.intervention-pair-plan/1"
                    and not pair_plan.get("unresolved")
                    and len(pair_plan.get("pairs",[]))
                    == completion.get("registered_pairs")
                    and hashlib.sha256(pair_plan_path.read_bytes()).hexdigest()
                    == completion.get("pair_plan_sha256")
                    and hashlib.sha256(source_completion_path.read_bytes()).hexdigest()
                    == completion.get("source_completion_sha256")
                )
                completion_valid=(
                    completion.get("successful") is True
                    and geometry_valid
                    and completion.get("negative_keep_probability")
                    == expected_negative_keep_probability
                    and completion.get("registered_pairs")
                    == completion.get("verified_pairs")
                    and completion.get("registered_pairs",0) > 0
                    and len(completion.get("pairs",[]))
                    == completion.get("registered_pairs")
                    and all(
                        row.get("valid") is True
                        and row.get("source_shards_verified") is True
                        and (
                            root / row.get("output","") / "manifest.json"
                        ).is_file()
                        and hashlib.sha256(
                            (root / row["output"] / "manifest.json").read_bytes()
                        ).hexdigest()
                        == row.get("manifest_sha256")
                        for row in completion.get("pairs",[])
                    )
                    and completion_index_valid
                )
            else:
                completion_valid=False
                completion_index_valid=False
        except (OSError,json.JSONDecodeError):
            completion_valid=False
            completion_index_valid=False
    sequence_geometry_compatible=manifest_lengths == {expected_sequence_length}
    if allow_local_tiling and len(manifest_lengths) == 1:
        source_length=next(iter(manifest_lengths))
        sequence_geometry_compatible=(
            isinstance(source_length,int)
            and source_length > expected_sequence_length
            and source_length % expected_sequence_length == 0
        ) or sequence_geometry_compatible
    geometry_compatible=(
        bool(paths)
        and missing_manifests == 0
        and sequence_geometry_compatible
        and negative_keep_probabilities == {expected_negative_keep_probability}
        and conditioning_compatible
    )
    compatible=geometry_compatible and completion_valid
    return {
        "glob": pattern,
        "files_now": len(paths),
        "bytes_now": sum(path.stat().st_size for path in paths),
        "available_now": bool(paths),
        "expected_sequence_length": expected_sequence_length,
        "manifest_sequence_lengths": sorted(
            value for value in manifest_lengths if isinstance(value,int)
        ),
        "expected_negative_keep_probability": expected_negative_keep_probability,
        "manifest_negative_keep_probabilities": sorted(
            value
            for value in negative_keep_probabilities
            if isinstance(value,(int,float))
        ),
        "missing_track_manifests": missing_manifests,
        "conditioning_lengths": {
            key: sorted(values) for key,values in conditioning_lengths.items()
        },
        "conditioning_compatible_now": conditioning_compatible,
        "zero_perturbation_padding_manifests": zero_perturbation_padding_manifests,
        "geometry_compatible_now": geometry_compatible,
        "local_tiling": (
            {
                "enabled": True,
                "tile_sequence_length": expected_sequence_length,
                "source_sequence_lengths": sorted(
                    value for value in manifest_lengths if isinstance(value,int)
                ),
            }
            if (
                allow_local_tiling
                and bool(paths)
                and manifest_lengths != {expected_sequence_length}
            )
            else None
        ),
        "completion_marker_required": require_completion_marker,
        "completion_marker": str(completion_path.relative_to(root)),
        "completion_marker_present": completion_path.is_file(),
        "completion_index_valid": completion_index_valid,
        "completion_marker_valid": completion_valid,
        "compatible_now": compatible,
    }

def _stage_patterns(value: str | Sequence[str]) -> list[str]:
    if isinstance(value,str):
        patterns=[value]
    elif isinstance(value,Sequence):
        patterns=list(value)
    else:
        raise ValueError("campaign stage data must be a glob or a list of globs")
    if not patterns or any(not isinstance(pattern,str) or not pattern for pattern in patterns):
        raise ValueError("campaign stage data globs must be nonempty strings")
    if len(set(patterns)) != len(patterns):
        raise ValueError("campaign stage data contains duplicate globs")
    return patterns

def _stage_data_summary(
    root: Path,
    value: str | Sequence[str],
    expected_sequence_length: int,
    expected_bin_size: int,
    expected_negative_keep_probability: float,
    *,
    require_completion_marker: bool,
    allow_local_tiling: bool,
    expected_conditioning_dimensions: dict[str,int] | None = None,
) -> dict:
    patterns=_stage_patterns(value)
    sources=[
        _glob_summary(
            root,
            pattern,
            expected_sequence_length,
            expected_bin_size,
            expected_negative_keep_probability,
            require_completion_marker=require_completion_marker,
            allow_local_tiling=allow_local_tiling,
            expected_conditioning_dimensions=expected_conditioning_dimensions,
        )
        for pattern in patterns
    ]
    return {
        "globs": patterns,
        "source_count": len(sources),
        "sources": sources,
        "files_now": sum(source["files_now"] for source in sources),
        "bytes_now": sum(source["bytes_now"] for source in sources),
        "available_now": all(source["available_now"] for source in sources),
        "compatible_now": all(source["compatible_now"] for source in sources),
    }

def build_campaign_plan(
    project_root: str | Path,
    campaign_path: str | Path,
    profile_config: str,
    out: str | Path,
) -> dict:
    root=Path(project_root).resolve()
    campaign_path=Path(campaign_path)
    if not campaign_path.is_absolute():
        campaign_path=root / campaign_path
    campaign=json.loads(campaign_path.read_text(encoding="utf-8"))
    if campaign.get("schema") != "pdac-circuit.chromatin-campaign/1":
        raise ValueError("invalid chromatin campaign schema")
    all_profiles=campaign["profiles"] + campaign.get("ablation_profiles",[])
    profile_rows=[row for row in all_profiles if row["config"] == profile_config]
    if len(profile_rows) != 1:
        raise ValueError(f"profile {profile_config!r} is not uniquely registered")
    minimum_free_vram_gb=profile_rows[0].get("minimum_free_vram_gb")
    if (
        isinstance(minimum_free_vram_gb,bool)
        or not isinstance(minimum_free_vram_gb,(int,float))
        or minimum_free_vram_gb <= 0
    ):
        raise ValueError("campaign profile must bind a positive minimum_free_vram_gb")
    minimum_free_vram_gb=float(minimum_free_vram_gb)
    config_path=root / profile_config
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    from .config import load_chromatin_config

    model_config,_,_ = load_chromatin_config(config_path)
    execution=campaign.get("execution",{})
    stage_data=execution.get("profile_stage_data",{}).get(
        profile_config,execution.get("stage_data",{})
    )
    if set(stage_data) != set(TRAINING_STAGES):
        raise ValueError("campaign execution must bind all four trainable stages")
    require_completion_markers=execution.get("require_complete_markers") is True
    compile_contracts=execution.get("stage_compile_contracts",{})
    if set(compile_contracts) != set(TRAINING_STAGES):
        raise ValueError("campaign execution must bind all four compile contracts")
    validation_studies=execution.get("stage_validation_studies",{})
    if set(validation_studies) != set(TRAINING_STAGES):
        raise ValueError("campaign execution must bind validation scope for all four stages")
    normalized_validation_studies={}
    for stage in TRAINING_STAGES:
        studies=validation_studies[stage]
        if studies is None:
            normalized_validation_studies[stage]=[]
            continue
        if (
            not isinstance(studies,list)
            or not studies
            or any(not isinstance(study,str) or not study for study in studies)
            or len(set(studies)) != len(studies)
        ):
            raise ValueError(
                f"campaign validation studies for {stage} must be null or a nonempty unique list"
            )
        normalized_validation_studies[stage]=[study.upper() for study in studies]
    if normalized_validation_studies["human_state_adaptation"] != ["GSE272463"]:
        raise ValueError(
            "human-state adaptation checkpoint selection must be bound only to GSE272463"
        )

    normalized_stage_data={
        stage: _stage_patterns(stage_data[stage]) for stage in TRAINING_STAGES
    }
    data={
        stage: _stage_data_summary(
            root,
            stage_data[stage],
            model_config.sequence_length,
            model_config.bin_size,
            float(compile_contracts[stage]["negative_keep_probability"]),
            require_completion_marker=require_completion_markers,
            allow_local_tiling=model_config.architecture
            == "direct_conditional_cnn",
            expected_conditioning_dimensions={
                "assay_features": model_config.assay_features,
                "state_features": model_config.state_features,
                "perturbation_features": model_config.perturbation_features,
            },
        )
        for stage in TRAINING_STAGES
    }
    checkpoint_root=root / execution.get("checkpoint_root","models/chromatin/campaign")
    profile_slug=Path(profile_config).stem
    reuse_healthy_from=profile_rows[0].get("reuse_healthy_from")
    if reuse_healthy_from and not any(
        row.get("config") == reuse_healthy_from for row in campaign["profiles"]
    ):
        raise ValueError("ablation healthy parent must be a primary campaign profile")
    stage_rows=list(enumerate(TRAINING_STAGES,start=1))
    if reuse_healthy_from:
        stage_rows=stage_rows[1:]
    nodes=[]
    for seed in campaign["seeds"]:
        previous=None
        external_parent=None
        if reuse_healthy_from:
            parent_slug=Path(reuse_healthy_from).stem
            parent_dir=(
                checkpoint_root / parent_slug / f"seed-{seed}" / "01-healthy_prior"
            )
            external_parent={
                "id": f"{parent_slug}.seed-{seed}.01-healthy_prior",
                "selected_checkpoint": str(
                    (parent_dir / "best.pt").relative_to(root)
                ),
            }
        for ordinal,stage in stage_rows:
            node_id=f"{profile_slug}.seed-{seed}.{ordinal:02d}-{stage}"
            checkpoint_dir=checkpoint_root / profile_slug / f"seed-{seed}" / f"{ordinal:02d}-{stage}"
            argv=[
                "pdac",
                "chromatin-train",
                "--config",
                profile_config,
            ]
            for pattern in normalized_stage_data[stage]:
                argv.extend(["--shards",pattern])
            argv.extend([
                "--checkpoint-dir",
                str(checkpoint_dir.relative_to(root)),
                "--stage",
                stage,
                "--seed",
                str(seed),
                "--min-free-vram-gb",
                str(minimum_free_vram_gb),
            ])
            for study in normalized_validation_studies[stage]:
                argv.extend(["--validation-study",study])
            initializer=previous or external_parent
            if initializer is not None:
                argv.extend(
                    ["--initialize-from",initializer["selected_checkpoint"]]
                )
            node={
                "id": node_id,
                "seed": seed,
                "ordinal": ordinal,
                "stage": stage,
                "minimum_free_vram_gb": minimum_free_vram_gb,
                "checkpoint_validation_studies": (
                    normalized_validation_studies[stage]
                    or "all_stage_validation_groups"
                ),
                "data": data[stage],
                "depends_on": [initializer["id"]] if initializer else [],
                "resume_checkpoint": str(
                    (checkpoint_dir / "latest.pt").relative_to(root)
                ),
                "selected_checkpoint": str(
                    (checkpoint_dir / "best.pt").relative_to(root)
                ),
                "argv": argv,
                "command": shlex.join(argv),
                "data_available_now": bool(data[stage]["available_now"]),
                "data_compatible_now": bool(data[stage]["compatible_now"]),
                "dependency_satisfied_now": (
                    initializer is None
                    or (root / initializer["selected_checkpoint"]).is_file()
                ),
            }
            node["runnable_now"]=(
                node["data_compatible_now"] and node["dependency_satisfied_now"]
            )
            nodes.append(node)
            previous=node

    payload={
        "schema": "pdac-circuit.chromatin-campaign-plan/1",
        "created_at": datetime.now().astimezone().isoformat(),
        "campaign": str(campaign_path.relative_to(root)),
        "campaign_sha256": hashlib.sha256(campaign_path.read_bytes()).hexdigest(),
        "profile": profile_rows[0],
        "profile_sha256": hashlib.sha256(config_path.read_bytes()).hexdigest(),
        "seeds": campaign["seeds"],
        "data": data,
        "nodes": nodes,
        "node_count": len(nodes),
        "reused_healthy_profile": reuse_healthy_from,
        "execution_policy": {
            "test_tuning_uses": campaign["selection"]["maximum_tuning_uses_of_test_surfaces"],
            "data_binding": execution["data_binding"],
            "resume_policy": execution["resume_policy"],
            "parallelism": execution["parallelism"],
            "stage_validation_studies": normalized_validation_studies,
        },
        "evaluation_stage": campaign["curriculum"][-1],
    }
    payload["plan_sha256"]=_json_hash(payload)
    destination=Path(out)
    if not destination.is_absolute():
        destination=root / destination
    destination.parent.mkdir(parents=True,exist_ok=True)
    temporary=destination.parent / f".{destination.name}.partial-{uuid.uuid4().hex}"
    temporary.write_text(json.dumps(payload,indent=2,sort_keys=True) + "\n",encoding="utf-8")
    temporary.replace(destination)
    return {
        "out": str(destination),
        "plan_sha256": payload["plan_sha256"],
        "nodes": len(nodes),
        "runnable_nodes_now": sum(node["runnable_now"] for node in nodes),
        "data": data,
    }
