from __future__ import annotations

import json
from pathlib import Path

from .config import load_chromatin_config
from .streaming import sha256_file

def verify_compiled_track(
    compiled_dir: str | Path,
    track_spec_path: str | Path,
    config_path: str | Path,
) -> dict:
    compiled_dir=Path(compiled_dir)
    spec=json.loads(Path(track_spec_path).read_text(encoding="utf-8"))
    model,_,_ = load_chromatin_config(config_path)
    manifest_path=compiled_dir / "manifest.json"
    failures=[]
    if not manifest_path.exists():
        return {"valid": False,"failures": ["manifest missing"]}
    manifest=json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != "pdac-circuit.chromatin-shards/3":
        failures.append("unsupported shard schema")
    track=manifest.get("track",{})
    for key in ("accession","source_sha256","genome","sample_group","split_role"):
        if track.get(key) != spec.get(key):
            failures.append(f"track {key} mismatch")
    expected_lengths={
        "assay_features": model.assay_features,
        "state_features": model.state_features,
        "perturbation_features": model.perturbation_features,
    }
    zero_perturbation_padding=None
    for key,expected in expected_lengths.items():
        observed=track.get(key,[])
        spec_values=spec.get(key,[])
        exact_zero_padding=(
            key == "perturbation_features"
            and 0 < len(observed) < expected
            and len(spec_values) == expected
            and not any(observed)
            and not any(spec_values)
        )
        if exact_zero_padding:
            zero_perturbation_padding={
                "from_features": len(observed),
                "to_features": expected,
                "required_all_zero": True,
            }
        elif len(observed) != expected:
            failures.append(f"{key} length mismatch")
    if manifest.get("sequence_length") != model.sequence_length:
        failures.append("sequence length mismatch")
    if manifest.get("bin_size") != model.bin_size:
        failures.append("bin size mismatch")
    if not manifest.get("native_genome_validated"):
        failures.append("native genome not validated")
    source_path=Path(str(spec.get("path") or ""))
    expected_source_hash=spec.get("source_sha256")
    source_verified=False
    if not source_path.is_file():
        failures.append("source bigWig missing")
    elif not expected_source_hash:
        failures.append("TrackSpec source sha256 missing")
    elif sha256_file(source_path) != expected_source_hash:
        failures.append("source bigWig sha256 mismatch")
    else:
        source_verified=True
    examples=0
    for shard in manifest.get("shards",[]):
        path=compiled_dir / shard.get("path","")
        if not path.is_file():
            failures.append(f"missing shard {path.name}")
            continue
        if path.stat().st_size != shard.get("bytes"):
            failures.append(f"shard byte mismatch {path.name}")
        expected_hash=shard.get("sha256")
        if not expected_hash or sha256_file(path) != expected_hash:
            failures.append(f"shard hash mismatch {path.name}")
        examples += int(shard.get("examples") or 0)
    if examples != int(manifest.get("windows_kept") or 0):
        failures.append("manifest example count mismatch")
    if examples < 1:
        failures.append("compiled track has no examples")
    return {
        "valid": not failures,
        "failures": failures,
        "examples": examples,
        "manifest_sha256": sha256_file(manifest_path),
        "source_sha256_verified": source_verified,
        "zero_perturbation_padding": zero_perturbation_padding,
    }

def audit_compiled_splits(shard_paths: list[str | Path]) -> dict:

    import numpy as np

    allowed_splits={
        "train",
        "validation",
        "locus_test",
        "state_test",
        "joint_locus_state_test",
        "external_study_test",
        "temporal_test",
    }
    counts: dict[str,int] = {}
    groups: dict[str,set[str]] = {}
    group_families: dict[str,set[str]] = {}
    studies: dict[str,set[str]] = {}
    train_intervals=set()
    held_locus_intervals=set()
    example_ids=set()
    duplicate_ids=[]
    failures=[]
    examples=0
    for value in sorted(Path(path) for path in shard_paths):
        with np.load(value,allow_pickle=False) as shard:
            required={
                "example_id",
                "split",
                "sample_group",
                "study",
                "genome",
                "chrom",
                "start",
                "end",
            }
            missing=sorted(required - set(shard.files))
            if missing:
                failures.append(f"{value.name} missing split-audit arrays: {missing}")
                continue
            n=len(shard["example_id"])
            for index in range(n):
                example_id=str(shard["example_id"][index])
                split=str(shard["split"][index])
                group=str(shard["sample_group"][index])
                study=str(shard["study"][index])
                if example_id in example_ids:
                    duplicate_ids.append(example_id)
                example_ids.add(example_id)
                if split not in allowed_splits:
                    failures.append(f"unsupported materialized split {split!r}")
                counts[split]=counts.get(split,0) + 1
                groups.setdefault(split,set()).add(group)
                studies.setdefault(split,set()).add(study)
                family=(
                    "held_state"
                    if split in {"state_test","joint_locus_state_test"}
                    else "external"
                    if split in {"external_study_test","temporal_test"}
                    else "development"
                )
                group_families.setdefault(group,set()).add(family)
                interval=(
                    str(shard["genome"][index]),
                    str(shard["chrom"][index]),
                    int(shard["start"][index]),
                    int(shard["end"][index]),
                )
                if split == "train":
                    train_intervals.add(interval)
                if split in {"locus_test","joint_locus_state_test"}:
                    held_locus_intervals.add(interval)
                examples += 1
    if duplicate_ids:
        failures.append(f"duplicate example IDs: {len(duplicate_ids)}")
    leaking_groups={
        group: sorted(families)
        for group,families in group_families.items()
        if "held_state" in families and "development" in families
    }
    if leaking_groups:
        failures.append(f"held-out state groups appear in development: {leaking_groups}")
    external_train_overlap=(
        studies.get("train",set())
        & (studies.get("external_study_test",set()) | studies.get("temporal_test",set()))
    ) - {""}
    if external_train_overlap:
        failures.append(
            f"external/temporal studies appear in train: {sorted(external_train_overlap)}"
        )
    interval_overlap=train_intervals & held_locus_intervals
    if interval_overlap:
        failures.append(f"held-out locus intervals appear in train: {len(interval_overlap)}")
    return {
        "schema": "pdac-circuit.compiled-split-audit/1",
        "ok": not failures,
        "failures": failures,
        "shards": len(shard_paths),
        "examples": examples,
        "unique_example_ids": len(example_ids),
        "counts": dict(sorted(counts.items())),
        "independent_groups": {
            split: len(values) for split,values in sorted(groups.items())
        },
        "studies": {split: sorted(values) for split,values in sorted(studies.items())},
        "train_held_locus_interval_overlap": len(interval_overlap),
        "held_state_development_group_overlap": len(leaking_groups),
    }
